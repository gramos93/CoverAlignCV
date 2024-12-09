import os
from typing import Tuple, Optional
import numpy as np
import pyrender
import trimesh
from PIL import Image
from dataclasses import dataclass


COUVERCLE_PATH = "../assets/3dmodels/couvercle.stl"
BOITIER_PATH = "../assets/3dmodels/boitier.stl"

OUTPUT_PATH = r"./output"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
RADIATEUR_WITH_MESH_PATH = "with_mesh.png"
RADIATEUR_WITHOUT_MESH_PATH = "without_mesh.png"

ORIGIN_OFFSET = np.array([651.86, 573.76, -2_894.40])
TOP_CAMERA_POSE = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 2.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
SIDE_CAMERA_POSE = np.array([
    [0.0, 0.0, -1.0, -2.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
TOP_LIGHT_POSE = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 10.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
SIDE_LIGHT_POSE = np.array([
    [0.0, 0.0, -1.0, -10.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.

    :param q1: First quaternion in form [x, y, z, w]
    :param q2: Second quaternion in form [x, y, z, w]
    :return: Result quaternion in form [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def random_perturbation(
    node: pyrender.Node,
    translation_range: float = 0.02,
    rotation_range: float = np.pi / 36,
) -> None:
    """
    Apply random translation and rotation to the pyrender node to simulate robotic imprecision.

    :param node: The pyrender Node to perturb
    :param translation_range: Max translation perturbation (in normalized units)
    :param rotation_range: Max rotation perturbation (in radians)
    """
    # Random translation
    translation = np.random.uniform(-translation_range, translation_range, size=3)
    if node.translation is None:
        node.translation = translation
    else:
        node.translation = node.translation + translation

    # Random rotation
    rotation_axis = np.random.normal(size=3)
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation_angle = np.random.uniform(-rotation_range, rotation_range)

    # Convert axis-angle to quaternion
    # Using half angle for quaternion calculation
    half_angle = rotation_angle / 2.0
    qx = rotation_axis[0] * np.sin(half_angle)
    qy = rotation_axis[1] * np.sin(half_angle)
    qz = rotation_axis[2] * np.sin(half_angle)
    qw = np.cos(half_angle)

    # Normalize quaternion
    quaternion = np.array([qx, qy, qz, qw])
    quaternion /= np.linalg.norm(quaternion)

    if node.rotation is None:
        node.rotation = quaternion
    else:
        # Combine with existing rotation using quaternion multiplication
        current_quat = node.rotation
        new_quat = quaternion_multiply(quaternion, current_quat)
        node.rotation = new_quat

@dataclass
class SceneConfig:
    viewport_width: int = 640
    viewport_height: int = 480
    bg_color: Tuple = (0.0, 0.0, 0.0, 0.0)
    light_intensity: float = 4.0
    light_color: Tuple = (1.0, 1.0, 1.0)
    camera_fov: float = np.pi / 3.0

class SceneHandler:
    def __init__(self, config: SceneConfig = SceneConfig()):
        self.config = config
        self.scene = pyrender.Scene(bg_color=config.bg_color)
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=config.viewport_width,
            viewport_height=config.viewport_height
        )

        self.cov_mesh_node: Optional[pyrender.Node] = None
        self.rad_mesh_node: Optional[pyrender.Node] = None
        self.camera_node: Optional[pyrender.Node] = None
        self.light_node: Optional[pyrender.Node] = None

        self.camera = pyrender.PerspectiveCamera(yfov=config.camera_fov)
        self.light = pyrender.DirectionalLight(
            color=config.light_color,
            intensity=config.light_intensity
        )

    @classmethod
    def from_stl_files(
        cls,
        mesh_path: str,
        rad_mesh_path: str,
        config: SceneConfig = SceneConfig()
    ) -> 'SceneHandler':
        handler = cls(config)
        handler.load_meshes(mesh_path, rad_mesh_path)
        return handler

    def load_meshes(self, mesh_path: str, rad_mesh_path: str) -> None:
        """Load and preprocess mesh files"""
        cov_mesh = trimesh.load(mesh_path)
        rad_mesh = trimesh.load(rad_mesh_path)

        # Preprocess meshes
        self._preprocess_meshes(cov_mesh, rad_mesh)
        self._create_pyrender_meshes(cov_mesh, rad_mesh)

    def _preprocess_meshes(self, cov_mesh: trimesh.Trimesh, rad_mesh: trimesh.Trimesh) -> None:
        """Apply transformations to meshes"""

        # Scale to unit box
        bounds = cov_mesh.bounds
        scale_factor = 1.0 / np.max(bounds[1] - bounds[0])

        for m in [cov_mesh, rad_mesh]:
            m.apply_translation(ORIGIN_OFFSET)
            m.apply_scale(scale_factor)

    def _create_pyrender_meshes(self, cov_mesh: trimesh.Trimesh, rad_mesh: trimesh.Trimesh) -> None:
        """Create pyrender meshes with materials"""
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1, 1, 1, 1.0],
            metallicFactor=0.1,
            roughnessFactor=0.1,
            smooth=False,
            alphaMode="OPAQUE"
        )
        pyrender_cov_mesh = pyrender.Mesh.from_trimesh(cov_mesh, material=material)
        pyrender_rad_mesh = pyrender.Mesh.from_trimesh(rad_mesh, material=material)

        self.cov_mesh_node = self.scene.add(pyrender_cov_mesh)
        self.rad_mesh_node = self.scene.add(pyrender_rad_mesh)

    def set_camera_pose(self, pose: np.ndarray) -> None:
        """Update camera position"""
        if self.camera_node:
            self.scene.remove_node(self.camera_node)
        self.camera_node = self.scene.add(self.camera, pose=pose)

    def set_light_pose(self, pose: np.ndarray) -> None:
        """Update light position"""
        if self.light_node:
            self.scene.remove_node(self.light_node)
        self.light_node = self.scene.add(self.light, pose=pose)

    def apply_random_perturbation(
        self,
        translation_range: float = 0.02,
        rotation_range: float = np.pi / 36,
    ):
        """Apply random perturbation to the meshes"""
        if self.cov_mesh_node:
            random_perturbation(self.cov_mesh_node, translation_range, rotation_range)

    def render(self, show_cov: bool = True) -> np.ndarray:
        """Render the scene"""
        if not show_cov and self.cov_mesh_node:
            self.scene.remove_node(self.cov_mesh_node)

        color, _ = self.renderer.render(self.scene)

        if not show_cov and self.cov_mesh_node:
            self.scene.add_node(self.cov_mesh_node)

        return color

    def cleanup(self):
        """Clean up resources"""
        self.renderer.delete()


def render_for_pose(pose="top") -> None:
        config = SceneConfig(
            viewport_width=640,
            viewport_height=480,
            light_intensity=4.0
        )
        handler = SceneHandler.from_stl_files(
            mesh_path=COUVERCLE_PATH,
            rad_mesh_path=BOITIER_PATH,
            config=config
        )
        if pose == "side":
            handler.set_camera_pose(SIDE_CAMERA_POSE)
            handler.set_light_pose(SIDE_LIGHT_POSE)
        else:
            handler.set_camera_pose(TOP_CAMERA_POSE)
            handler.set_light_pose(TOP_LIGHT_POSE)

        with_mesh_path = f"{OUTPUT_PATH}/{pose}_{RADIATEUR_WITH_MESH_PATH}"
        without_mesh_path = f"{OUTPUT_PATH}/{pose}_{RADIATEUR_WITHOUT_MESH_PATH}"

        with_cov = handler.render(show_cov=True)
        without_cov = handler.render(show_cov=False)
        handler.cleanup()

        Image.fromarray(with_cov).save(with_mesh_path)
        Image.fromarray(without_cov).save(without_mesh_path)


if __name__ == "__main__":
    render_for_pose("top")
    render_for_pose("side")
