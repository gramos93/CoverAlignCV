import numpy as np
import cv2
from dataclasses import dataclass
from typing import Sequence



@dataclass
class OpenCVImage:
    np_image: np.ndarray
    cv_image: np.ndarray
    gray: np.ndarray


def image_preprocessing(image_path: str) -> OpenCVImage:
    """Convert an array image from RGB to BGR format and then to grayscale"""
    np_image = np.array(image_path)
    cv_image = cv2.imread(image_path)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    return OpenCVImage(
        np_image=np_image,
        cv_image=cv_image,
        gray=gray
    )


@dataclass
class FindTheEdges:
    contours: Sequence[np.ndarray]
    image: np.ndarray


def detect_cover(processed_image: OpenCVImage) -> FindTheEdges:
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
    result_edges: FindTheEdges
) -> FindTheEdges:
    # Create the mask
    mask = np.zeros_like(processed_image.gray)
    cv2.drawContours(mask, result_edges.contours, -1, (255, 0, 0), thickness=cv2.FILLED)

    # Find the contours
    inverted_image = cv2.bitwise_not(processed_image.gray, mask=mask)
    _, thresh = cv2.threshold(inverted_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the edges
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
    # Detects circle shaped holes
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
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    else:
        print("Aucun cercle détecté.")

    return FindTheCercles(
        circles=circles,
        centers=centers,
        image=image
    )


def display(image, title="Detected Holes"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Preprocessing
    img_path = r"top_view.png"
    img = image_preprocessing(img_path)

    # Detect cover edges
    cover_result = detect_cover(img)
    display(cover_result.image, "Detects the cover")
    cv2.imwrite(str("cover_edges_" + img_path), cover_result.image)

    # Detect hole edges
    hole_result = detect_holes_in_cover(img, cover_result)
    display(hole_result.image, "Detects the holes")
    cv2.imwrite(str("hole_edges_" + img_path), hole_result.image)

    # Detect cercle edges
    cercle_result = detect_cercles(img)
    display(cercle_result.image, "Detects the cercles")
    cv2.imwrite(str("circle_edges_" + img_path), cercle_result.image)