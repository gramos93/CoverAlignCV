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
    vertical_line_zone: Tuple[int, int, int, int] = (170, 130, 30, 250)
    horizontal_line_zone: Tuple[int, int, int, int] = (170, 130, 175, 45)
    hole_zone: Tuple[int, int, int, int] = (312, 231, 35, 35)


class RadiatorHandler:
    def __init__(self, image: OpenCVImage, config: RadDetConfig = RadDetConfig()):
        self.config = config
        self.image = image
        self.processed_image = self.image.cv_image.copy()
        self._intersection: Optional[Tuple[int, int]] = None
        self._hole: Optional[Tuple[int, int]] = None

    def _detect_line(self, zone: Tuple[int, int, int, int], vertical: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """Detect either vertical or horizontal line in the specified zone"""
        x, y, w, h = zone
        cv2.rectangle(self.processed_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # blurred = cv2.GaussianBlur(self.image.gray[y:y + h, x:x + w], (5, 5), 0)
        blurred = self.image.gray[y:y + h, x:x + w]
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is None:
            return None
        else:
            lines = lines.reshape(-1, 4)

        lines = sorted(
            lines,
            key=lambda x: min(x[1], x[3]) if not vertical else min(x[0], x[2]),
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                if vertical and abs(x1 - x2) < 5:
                    return (x1 + x, y1 + y, x2 + x, y2 + y)

                elif not vertical and abs(y1 - y2) < 5:
                    return (x1 + x, y1 + y, x2 + x, y2 + y)

        return None

    def _detect_hole(self, zone: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int]]:
        """Detect hole in the specified zone"""
        x, y, w, h = zone
        cv2.rectangle(self.processed_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # NOTE: In the simulation the blurring is not helping.
        # blurred = cv2.GaussianBlur(self.image.gray[y:y + h, x:x + w], (5, 5), 0)
        circles = cv2.HoughCircles(
            self.image.gray[y:y + h, x:x + w],
            cv2.HOUGH_GRADIENT,
            dp=h/8,
            minDist=20,
            param1=100,
            param2=30,
            minRadius=0,
            maxRadius=0
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            x_c, y_c, r = circles[0][0]
            return (x_c + x, y_c + y, r)
        return None

    def process_image(self) -> None:
        """Process the image to detect lines and hole"""
        vertical_line = self._detect_line(self.config.vertical_line_zone, vertical=True)
        horizontal_line = self._detect_line(self.config.horizontal_line_zone, vertical=False)
        self._hole = self._detect_hole(self.config.hole_zone)

        if all([vertical_line, horizontal_line, self._hole]):
            x1_v, y1_v, x2_v, y2_v = vertical_line
            x1_h, y1_h, x2_h, y2_h = horizontal_line
            self._intersection = self.calculate_intersection(vertical_line, horizontal_line)
            cx, cy, r  = self._hole

            # Draw results
            cv2.line(self.processed_image, (x1_v, y1_v), (x2_v, y2_v), (255, 0, 255), 2)
            cv2.line(self.processed_image, (x1_h, y1_h), (x2_h, y2_h), (255, 0, 255), 2)
            cv2.circle(self.processed_image, (cx, cy), r, (255, 0, 0), 1)

            if self._intersection is not None:
                cv2.circle(self.processed_image, self._intersection, r, (0, 255, 0), 1)

        else:
            print("[RadiatorHandler] Lines or hole not found")

    def calculate_intersection(self, vertical_line, horizontal_line) -> Optional[Tuple[int, int]]:
        """Return the intersection point of vertical and horizontal lines"""

        x1, y1, x2, y2 = vertical_line
        x3, y3, x4, y4 = horizontal_line

        denom = ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
        if denom == 0:
            return None

        x = (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4) / denom
        y = (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4) / denom

        return (int(x), int(y))

    def get_hole(self) -> Optional[Tuple[int, int]]:
        """Return the center coordinates of the detected hole"""
        return self._hole

    def display_result(self, img: Optional[np.ndarray] = None) -> None:
        """Display the processed image with detections"""
        RESIZE_MAX_HEIGHT = 800
        RESIZE_MAX_WIDTH = 1200

        img = img if (img is not None) else self.processed_image
        height, width = img.shape[:2]

        if height > RESIZE_MAX_HEIGHT or width > RESIZE_MAX_WIDTH:
            scale = min(
                RESIZE_MAX_HEIGHT / height,
                RESIZE_MAX_WIDTH / width
            )
            dim = (int(width * scale), int(height * scale))
            resized = cv2.resize(img, dim)
            cv2.imshow("Detection Result", resized)
        else:
            cv2.imshow("Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

@dataclass
class CoverImage:
    top: np.ndarray
    side: np.ndarray

@dataclass
class CoverDetConfig:
    holes_zone: Tuple[int, int, int, int] = (170, 220, 180, 50)
    edge_zone: Tuple[int, int, int, int] = (130, 245, 220, 25)
    min_hole_distance: int = 100
    max_hole_distance: int = 400

class CoverHandler:
    def __init__(self, image_top: OpenCVImage, image_side: OpenCVImage, config: CoverDetConfig = CoverDetConfig()):
        self.config = config
        self.image_top = image_top
        self.image_side = image_side
        self.processed_image: CoverImage = CoverImage(
            top=self.image_top.cv_image.copy(),
            side=self.image_side.cv_image.copy(),
        )
        self._side_edge : Optional[Tuple[int, int, int, int]] = None
        self._left_hole: Optional[Tuple[int, int]] = None
        self._right_hole: Optional[Tuple[int, int]] = None
        self._y_angle: Optional[float] = None
        self._x_angle: Optional[float] = None

    def _detect_line(self, zone: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Detect slightly rotated horizontal line in the specified zone

        Args:
            zone: Tuple of (x, y, width, height) defining search area

        Returns:
            Optional[Tuple[int, int, int, int]]: Coordinates of shortest near-horizontal line found,
            or None if no suitable line detected
        """
        x, y, w, h = zone
        cv2.rectangle(self.processed_image.side, (x, y), (x+w, y+h), (0, 255, 0), 2)

        blurred = cv2.GaussianBlur(self.image_side.gray[y:y + h, x:x + w], (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

        if lines is not None:
            shortest_line = None
            min_length = float('inf')

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate angle with horizontal axis
                angle = self._calculate_angle((x1, y1, x2, y2))

                # Check if line is within ±5 degrees of horizontal
                if abs(angle) <= 5:
                    # Calculate line length
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                    # Update shortest line if this is shorter
                    if length < min_length:
                        min_length = length
                        shortest_line = (x1 + x, y1 + y, x2 + x, y2 + y)

            return shortest_line
        return None

    def _calculate_angle(self, line: Tuple[int, int, int, int]) -> float:
        """Calculate angle between line and horizontal axis

        Args:
            line: Tuple of (x1, y1, x2, y2) coordinates representing line endpoints

        Returns:
            float: Angle in degrees between the line and horizontal axis
        """
        if line is None:
            return None

        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        return np.degrees(np.arctan2(dy, dx))

    def _detect_holes(self, zone: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Detect holes in the specified zone"""
        x, y, w, h = zone
        cv2.rectangle(self.processed_image.top, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # NOTE: See RadiatorHandler for the blurring note.
        # blurred = cv2.GaussianBlur(self.image_top.gray[y:y + h, x:x + w], (5, 5), 0)
        circles = cv2.HoughCircles(
            self.image_top.gray[y:y + h, x:x + w],
            cv2.HOUGH_GRADIENT,
            dp=h/16,
            minDist=20,
            param1=100,
            param2=30,
            minRadius=0,
            maxRadius=10,
        )

        holes = []
        if circles is not None:
            circles = np.around(circles)
            for circle in circles[0]:
                x_c, y_c, r = circle
                center = (int(x_c + x), int(y_c + y))
                holes.append(center)
                cv2.circle(self.processed_image.top, center, int(r), (0, 255, 255), 1)

        return holes

    def process_image(self) -> None:
        """Process the image to detect holes and calculate angle"""
        holes = self._detect_holes(self.config.holes_zone)
        self._side_edge = self._detect_line(self.config.edge_zone)
        if self._side_edge is None:
            self._x_angle = 0
            print("[CoverHandler] Side edge was not detected, not correcting for X deviation.")
        else:
            cv2.line(
                img=self.processed_image.side,
                pt1=(self._side_edge[0], self._side_edge[1]),
                pt2=(self._side_edge[2], self._side_edge[3]),
                color=(255, 0, 255),
                thickness=2
            )
            self._x_angle = self._calculate_angle(self._side_edge)

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
                self._y_angle = self._calculate_angle((*self._left_hole, *self._right_hole))

                # Draw the line connecting holes
                cv2.line(
                    img=self.processed_image.top,
                    pt1=self._left_hole,
                    pt2=self._right_hole,
                    color=(255, 0, 255),
                    thickness=2
                )
            else:
                print("[CoverHandler] Holes distance out of range.")
        else:
            print("[CoverHandler] Not enough holes detected.")

    def get_left_hole(self) -> Optional[Tuple[int, int]]:
        """Return the coordinates of the left hole"""
        return self._left_hole

    def get_right_hole(self) -> Optional[Tuple[int, int]]:
        """Return the coordinates of the right hole"""
        return self._right_hole

    def get_angle(self) -> Optional[float]:
        """Return the angle between holes line and horizontal axis"""
        return self._y_angle

    def display_result(self) -> None:
        """Display the processed image with detections"""
        RESIZE_MAX_HEIGHT = 800
        RESIZE_MAX_WIDTH = 1200

        height, width = self.processed_image.top.shape[:2]
        if height > RESIZE_MAX_HEIGHT or width > RESIZE_MAX_WIDTH:
            scale = min(
                RESIZE_MAX_HEIGHT / height,
                RESIZE_MAX_WIDTH / width
            )
            dim = (int(width * scale), int(height * scale))
            resized_top = cv2.resize(self.processed_image.top, dim)
            resized_side = cv2.resize(self.processed_image.side, dim)
            cv2.imshow("Cover Detection Top Result", resized_top)
            cv2.imshow("Cover Detection Side Result", resized_side)
        else:
            cv2.imshow("Cover Detection Top Result", self.processed_image.top)
            cv2.imshow("Cover Detection Side Result", self.processed_image.side)
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
    circles = cv2.HoughCircles(masked_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=20, minRadius=2, maxRadius=10)
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

    from preprocessing import create_open_cv_image
    from render import (
        SceneHandler,
        COUVERCLE_PATH,
        BOITIER_PATH,
        TOP_CAMERA_POSE,
        SIDE_CAMERA_POSE,
        TOP_LIGHT_POSE,
        SIDE_LIGHT_POSE,
    )
    scene = SceneHandler.from_stl_files(COUVERCLE_PATH, BOITIER_PATH)
    scene.set_camera_pose(TOP_CAMERA_POSE)
    scene.set_light_pose(TOP_LIGHT_POSE)

    rad_img = scene.render(show_cov=False)
    rad_img = create_open_cv_image(rad_img)
    rad_handler = RadiatorHandler(rad_img)

    rad_handler.process_image()
    rad_handler.display_result()

    cover_img_top = scene.render(show_cov=True)
    cover_img_top = create_open_cv_image(cover_img_top)

    scene.set_camera_pose(SIDE_CAMERA_POSE)
    scene.set_light_pose(SIDE_LIGHT_POSE)
    cover_img_side = scene.render(show_cov=True)
    cover_img_side = create_open_cv_image(cover_img_side)

    cover_handler = CoverHandler(cover_img_top, cover_img_side)
    cover_handler.process_image()
    cover_handler.display_result()
