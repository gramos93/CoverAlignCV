import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from render import (
    OUTPUT_PATH,
    RADIATEUR_WITH_MESH_PATH,
    RADIATEUR_WITHOUT_MESH_PATH
)



@dataclass
class OpenCVImage:
    np_image: np.ndarray
    cv_image: np.ndarray
    gray: np.ndarray

def create_open_cv_image(image: np.ndarray) -> OpenCVImage:
    return OpenCVImage(
        np_image=image,
        cv_image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    )

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


def plot_histogram(image):
    hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))
    lower_threshold = 50
    non_zero_hist = hist[lower_threshold: ]
    non_zero_bins = bins[lower_threshold+1:]

    plt.bar(non_zero_bins, non_zero_hist, width=1, color='black')
    plt.title('Histogramme de l\'image (valeurs > 0)')
    plt.xlabel('Intensité des pixels')
    plt.ylabel('Fréquence')
    plt.show()


def display(image, title="Detected Holes"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pose = "top"
    RADIATEUR_WITH_MESH_PATH = f"{OUTPUT_PATH}/{pose}_{RADIATEUR_WITH_MESH_PATH}"
    RADIATEUR_WITHOUT_MESH_PATH = f"{OUTPUT_PATH}/{pose}_{RADIATEUR_WITHOUT_MESH_PATH}"

    # Charger l'image
    image_with_mesh = cv2.imread(RADIATEUR_WITH_MESH_PATH, cv2.IMREAD_GRAYSCALE)
    image_without_mesh = cv2.imread(RADIATEUR_WITHOUT_MESH_PATH, cv2.IMREAD_GRAYSCALE)

    plot_histogram(image_with_mesh)
    plot_histogram(image_without_mesh)
