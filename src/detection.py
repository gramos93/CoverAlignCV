import numpy as np
import cv2
from dataclasses import dataclass
from typing import Sequence
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


def define_and_mask_area(image, x=73, y=142, w=467, h=224) -> tuple[
    tuple[int, int, int, int], tuple[int, int, int, int]]:
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
