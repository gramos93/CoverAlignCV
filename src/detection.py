import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence, List
from preprocessing import (
    display,
    image_preprocessing,
    OpenCVImage
)
from render import (
    OUTPUT_PATH,
    RADIATEUR_WITH_MESH_PATH,
    RADIATEUR_WITHOUT_MESH_PATH,
)


@dataclass
class RadDetConfig:
    vertical_line_zone: Tuple[int, int, int, int] = (130, 500, 90, 700)
    horizontal_line_zone: Tuple[int, int, int, int] = (190, 1100, 800, 100)
    hole_zone: Tuple[int, int, int, int] = (500, 550, 125, 125)


class RadiatorHandler:
    def __init__(self, image: OpenCVImage, config: RadDetConfig = RadDetConfig()):
        self.config = config
        self.image = image
        self.processed_image = self.image.cv_image.copy()
        self._intersection: Optional[Tuple[int, int]] = None
        self._hole_center: Optional[Tuple[int, int]] = None

    def _detect_line(self, zone: Tuple[int, int, int, int], vertical: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """Detect either vertical or horizontal line in the specified zone"""
        x, y, w, h = zone
        sub_image = self.image.cv_image[y:y + h, x:x + w]
        gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if vertical and abs(x1 - x2) < 10:  # Vertical line
                    return (x1 + x, y1 + y, x2 + x, y2 + y)
                elif not vertical and abs(y1 - y2) < 10:  # Horizontal line
                    return (x1 + x, y1 + y, x2 + x, y2 + y)
        return None

    def _detect_hole(self, zone: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """Detect hole in the specified zone"""
        x, y, w, h = zone
        sub_image = self.image.cv_image[y:y + h, x:x + w]
        gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=20,
            minRadius=1,
            maxRadius=12
        )

        if circles is not None:
            circles = np.around(circles)
            x_c, y_c, r = circles[0][0]
            center = (x_c + x, y_c + y)
            cv2.circle(self.processed_image, center, r, (0, 255, 255), 2)
            return center
        return None

    def process_image(self) -> None:
        """Process the image to detect lines and hole"""
        vertical_line = self._detect_line(self.config.vertical_line_zone, vertical=True)
        horizontal_line = self._detect_line(self.config.horizontal_line_zone, vertical=False)
        self._hole_center = self._detect_hole(self.config.hole_zone)

        if all([vertical_line, horizontal_line, self._hole_center]):
            x1_v, y1_v, x2_v, y2_v = vertical_line
            x1_h, y1_h, x2_h, y2_h = horizontal_line
            self._intersection = (x1_v, y1_h)

            # Draw results
            cv2.line(self.processed_image, (x1_v, y1_v), (x2_v, y2_v), (255, 0, 255), 3)
            cv2.line(self.processed_image, (x1_h, y1_h), (x2_h, y2_h), (255, 0, 255), 3)
            cv2.circle(self.processed_image, self._intersection, 5, (255, 0, 0), -1)

    def get_intersection(self) -> Optional[Tuple[int, int]]:
        """Return the intersection point of vertical and horizontal lines"""
        return self._intersection

    def get_hole_center(self) -> Optional[Tuple[int, int]]:
        """Return the center coordinates of the detected hole"""
        return self._hole_center

    def get_relative_coordinates(self) -> Optional[Tuple[int, int]]:
        """Return the relative coordinates of the hole center to the intersection"""
        if self._intersection and self._hole_center:
            return (
                self._hole_center[0] - self._intersection[0],
                self._hole_center[1] - self._intersection[1]
            )
        return None

    def display_result(self) -> None:
        """Display the processed image with detections"""
        RESIZE_MAX_HEIGHT = 800
        RESIZE_MAX_WIDTH = 1200

        height, width = self.processed_image.shape[:2]
        if height > RESIZE_MAX_HEIGHT or width > RESIZE_MAX_WIDTH:
            scale = min(
                RESIZE_MAX_HEIGHT / height,
                RESIZE_MAX_WIDTH / width
            )
            dim = (int(width * scale), int(height * scale))
            resized = cv2.resize(self.processed_image, dim)
            cv2.imshow("Detection Result", resized)
        else:
            cv2.imshow("Detection Result", self.processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

@dataclass
class CoverDetConfig:
    # Define the zone where we expect to find the cover holes
    holes_zone: Tuple[int, int, int, int] = (200, 200, 600, 200)  # (x, y, width, height)
    min_hole_distance: int = 100  # Minimum distance between holes
    max_hole_distance: int = 400  # Maximum distance between holes

class CoverHandler:
    def __init__(self, image: OpenCVImage, config: CoverDetConfig = CoverDetConfig()):
        self.config = config
        self.image = image
        self.processed_image = self.image.cv_image.copy()
        self._left_hole: Optional[Tuple[int, int]] = None
        self._right_hole: Optional[Tuple[int, int]] = None
        self._angle: Optional[float] = None

    def _detect_holes(self, zone: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Detect holes in the specified zone"""
        x, y, w, h = zone
        sub_image = self.image.cv_image[y:y + h, x:x + w]
        gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)

        # Apply image processing to enhance hole detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self.config.min_hole_distance,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=20
        )

        holes = []
        if circles is not None:
            circles = np.around(circles)
            for circle in circles[0]:
                x_c, y_c, r = circle
                center = (int(x_c + x), int(y_c + y))
                holes.append(center)
                cv2.circle(self.processed_image, center, int(r), (0, 255, 255), 2)

        return holes

    def _calculate_angle(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate angle between line and horizontal axis"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.degrees(np.arctan2(dy, dx))

    def process_image(self) -> None:
        """Process the image to detect holes and calculate angle"""
        holes = self._detect_holes(self.config.holes_zone)

        if len(holes) >= 2:
            # Sort holes by x-coordinate to get left and right
            holes.sort(key=lambda x: x[0])

            # Take the leftmost and rightmost holes
            self._left_hole = holes[0]
            self._right_hole = holes[-1]

            # Calculate distance between holes
            distance = np.sqrt((self._right_hole[0] - self._left_hole[0])**2 +
                             (self._right_hole[1] - self._left_hole[1])**2)

            # Check if distance is within acceptable range
            if self.config.min_hole_distance <= distance <= self.config.max_hole_distance:
                self._angle = self._calculate_angle(self._left_hole, self._right_hole)

                # Draw the line connecting holes
                cv2.line(
                    img=self.processed_image,
                    pt1=self._left_hole,
                    pt2=self._right_hole,
                    color=(255, 0, 255),
                    thickness=2
                )

    def get_left_hole(self) -> Optional[Tuple[int, int]]:
        """Return the coordinates of the left hole"""
        return self._left_hole

    def get_right_hole(self) -> Optional[Tuple[int, int]]:
        """Return the coordinates of the right hole"""
        return self._right_hole

    def get_angle(self) -> Optional[float]:
        """Return the angle between holes line and horizontal axis"""
        return self._angle

    def display_result(self) -> None:
        """Display the processed image with detections"""
        RESIZE_MAX_HEIGHT = 800
        RESIZE_MAX_WIDTH = 1200

        height, width = self.processed_image.shape[:2]
        if height > RESIZE_MAX_HEIGHT or width > RESIZE_MAX_WIDTH:
            scale = min(
                RESIZE_MAX_HEIGHT / height,
                RESIZE_MAX_WIDTH / width
            )
            dim = (int(width * scale), int(height * scale))
            resized = cv2.resize(self.processed_image, dim)
            cv2.imshow("Cover Detection Result", resized)
        else:
            cv2.imshow("Cover Detection Result", self.processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

top_left_corner_of_radiator = (174, 143)

@dataclass
class FindTheEdges:
    contours: Sequence[np.ndarray]
    image: np.ndarray


def detect_cover(processed_image: OpenCVImage) -> FindTheEdges:
    """Detect and draw the radiator edges in the image."""
    # Find the contours
    inverted_image = cv2.bitwise_not(processed_image.gray)
    _, thresh = cv2.threshold(inverted_image, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours
    image = np.copy(processed_image.cv_image)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    return FindTheEdges(
        contours=contours,
        image=image
    )


def detect_holes_in_cover(
    processed_image: OpenCVImage,
    result_cover: FindTheEdges
) -> FindTheEdges:
    """Detect and draw the holes edges from the radiator area."""
    # Create the mask
    mask = np.zeros_like(processed_image.gray)
    cv2.drawContours(mask, result_cover.contours, -1, (255, 0, 0), thickness=cv2.FILLED)

    # Find the contours
    inverted_image = cv2.bitwise_not(processed_image.gray, mask=mask)
    _, thresh = cv2.threshold(inverted_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours
    image = np.copy(processed_image.cv_image)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 1)

    return FindTheEdges(
        contours=contours,
        image=image
    )


@dataclass
class FindTheCercles:
    circles: Sequence[np.ndarray]
    centers: list[tuple[int, int, int]]
    image: np.ndarray


def detect_cercles(processed_image: OpenCVImage) -> FindTheCercles:
    """Detect and draw the holes in the image."""
    # Detect the circle shaped holes
    circles = cv2.HoughCircles(
        processed_image.gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=20, minRadius=5, maxRadius=10
    )
    # Draw circles shaped holes
    centers = []
    image = np.copy(processed_image.cv_image)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            centers.append((x, y, r))
            cv2.circle(image, (x, y), r, (255, 0, 0), 2)
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    else:
        print("Aucun cercle détecté.")

    return FindTheCercles(
        circles=circles,
        centers=centers,
        image=image
    )


@dataclass
class FindTheRectangle:
    rectangles: list[tuple[int, int, int, int]]
    centers: list[tuple[int, int]]
    image: np.ndarray


def detect_rectangle(processed_image: OpenCVImage) -> FindTheRectangle:
    """Detect and draw the rectangle around the radiator"""
    new_image = np.copy(processed_image.cv_image)
    _, bin_image = cv2.threshold(processed_image.gray, 127, 255, cv2.THRESH_BINARY)
    radiator_contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles, centers = [], []
    for contour in radiator_contours:
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append((x, y, w, h))
        centers.append((x + w // 2, y + h // 2))
        cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return FindTheRectangle(
        rectangles=rectangles,
        centers=centers,
        image=new_image
    )


def detect_cercles_in_cover_area(processed_image: OpenCVImage) -> FindTheCercles:
    """Detect and draw the holes from the cover area."""
    new_image = np.copy(processed_image.cv_image)

    # Define the cover area
    cover_area, cover_mask = define_and_mask_area(processed_image.gray, w=170+150)
    x, y, w, h = cover_area
    cv2.rectangle(new_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    masked_image = cv2.bitwise_and(processed_image.gray, processed_image.gray, mask=cover_mask)

    # Detect and draw the holes from the cover area
    circles = cv2.HoughCircles(masked_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=20, minRadius=5, maxRadius=10)
    centers = []
    if circles is not None:
        print("Cercle détecté.")
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r)  in circles:
            centers.append((x, y, r))
            cv2.circle(new_image, (x, y), r, (255, 0, 0), 2)
            cv2.circle(new_image, (x, y), 1, (0, 0, 255), -1)
    else:
        print("Aucun cercle détecté.")

    return FindTheCercles(
        circles=circles,
        centers=centers,
        image=new_image
    )


def define_and_mask_area(image, x=73, y=142, w=467, h=224):
    """Define an area of the image and its mask."""
    area = (x, y, w, h)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    return area, mask



if __name__ == "__main__":
    pose = "top"
    RADIATEUR_WITH_MESH_PATH = f"{OUTPUT_PATH}/{pose}_{RADIATEUR_WITH_MESH_PATH}"
    RADIATEUR_WITHOUT_MESH_PATH = f"{OUTPUT_PATH}/{pose}_{RADIATEUR_WITHOUT_MESH_PATH}"

    # Preprocessing
    img = image_preprocessing(RADIATEUR_WITH_MESH_PATH)

    # Detect cover edges
    cover_result = detect_cover(img)
    display(cover_result.image, "Detects the cover")

    # Detect hole edges
    hole_result = detect_holes_in_cover(img, cover_result)
    display(hole_result.image, "Detects the holes")

    # Detect cercle edges
    cercle_result = detect_cercles(img)
    display(cercle_result.image, "Detects the cercles")

    # Detect and draw the rectangle around the radiator
    rectangle_result = detect_rectangle(img)
    display(rectangle_result.image, "Detects the rectangle")

    cercles_in_rectangle_result = detect_cercles_in_cover_area(img)
    display(cercles_in_rectangle_result.image, "Detects the holes in the cover")
