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

RAD_ORIGIN_OFFSET = np.array([651.86, 573.76, -2_894.40])
COV_ORIGIN_OFFSET = np.array([-643.50, -571.50, 2_894.0])
TOP_CAMERA_POSE = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
SIDE_CAMERA_POSE = np.array(
    [
        [0.0, 0.0, -1.0, -2.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
TOP_LIGHT_POSE = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 10.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
SIDE_LIGHT_POSE = np.array(
    [
        [0.0, 0.0, -1.0, -10.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.

    :param q1: First quaternion in form [x, y, z, w]
    :param q2: Second quaternion in form [x, y, z, w]
    :return: Result quaternion in form [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quaternion_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate a vector v by a quaternion q.

    :param v: 3D vector to rotate
    :param q: quaternion [x,y,z,w]
    :return: rotated vector
    """
    # Convert vector to pure quaternion
    v_quat = np.array([v[0], v[1], v[2], 0.0])

    # Compute q * v * q^-1
    q_inv = np.array([-q[0], -q[1], -q[2], q[3]])  # conjugate
    temp = quaternion_multiply(q, v_quat)
    rotated = quaternion_multiply(temp, q_inv)

    return rotated[0:3]  # Return just the vector part


class RotationAxis:
    """Defines a rotation axis in 3D space using a point and direction"""
    point = np.array([0.0, 0.0, -1.0])  # Point that the axis passes through
    direction = np.array([1.0, 0.0, 0.0])  # Direction vector of the axis (+X)


@dataclass
class Perturbation:
    """Represents a sampled perturbation with translation and rotation"""
    translation: np.ndarray
    rotation: np.ndarray  # quaternion
    rotation_point: np.ndarray  # point around which to rotate

    def __repr__(self) -> str:
        """
        Returns a string representation of the perturbation showing:
        - Translation in x,y,z
        - Rotation in quaternion format [x,y,z,w]
        - Rotation point in x,y,z
        """
        return (f"Perturbation(\n"
                f"  translation: [{self.translation[0]:.4f}, {self.translation[1]:.4f}, {self.translation[2]:.4f}],\n"
                f"  rotation: [{self.rotation[0]:.4f}, {self.rotation[1]:.4f}, {self.rotation[2]:.4f}, {self.rotation[3]:.4f}],\n"
                f"  rotation_point: [{self.rotation_point[0]:.4f}, {self.rotation_point[1]:.4f}, {self.rotation_point[2]:.4f}]\n"
                f")")


@dataclass
class PerturbationConfig:
    # Translation ranges per axis (None to disable perturbation on that axis)
    translation_x_range: Optional[Tuple[float, float]] = (-0.1, -0.02)
    translation_y_range: Optional[Tuple[float, float]] = None
    translation_z_range: Optional[Tuple[float, float]] = None

    # Rotation configuration
    rotation_axis: Optional[RotationAxis] = RotationAxis()  # Default X axis through origin
    rotation_range: Optional[Tuple[float, float]] = (-0.0872665/4, -0.0872665/2)  # 1.25 to 2.5 deg


def sample_perturbation(config: PerturbationConfig) -> Perturbation:
    """
    Sample a random perturbation based on the configuration.

    :param config: PerturbationConfig object containing the ranges and rotation axis
    :return: Perturbation object containing the sampled translation and rotation
    """
    # Sample random translation
    translation = np.zeros(3)
    for i, range_tuple in enumerate([config.translation_x_range,
                                   config.translation_y_range,
                                   config.translation_z_range]):
        if range_tuple is not None:
            min_range, max_range = range_tuple
            translation[i] = np.random.uniform(min_range, max_range)

    # Sample random rotation
    if config.rotation_axis is not None and config.rotation_range is not None:
        # Normalize rotation axis direction
        rotation_direction = config.rotation_axis.direction / np.linalg.norm(config.rotation_axis.direction)

        # Random angle within range
        min_range, max_range = config.rotation_range
        angle = np.random.uniform(min_range, max_range)

        # Convert to quaternion
        half_angle = angle / 2.0
        qx = rotation_direction[0] * np.sin(half_angle)
        qy = rotation_direction[1] * np.sin(half_angle)
        qz = rotation_direction[2] * np.sin(half_angle)
        qw = np.cos(half_angle)

        # Create and normalize quaternion
        quaternion = np.array([qx, qy, qz, qw])
        quaternion /= np.linalg.norm(quaternion)

        rotation_point = config.rotation_axis.point
    else:
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        rotation_point = np.zeros(3)

    return Perturbation(translation=translation,
                       rotation=quaternion,
                       rotation_point=rotation_point)

def apply_perturbation(node: pyrender.Node, perturbation: Perturbation) -> None:
    """
    Apply a perturbation to a pyrender node.

    :param node: The pyrender Node to perturb
    :param perturbation: Perturbation object containing the transformation to apply
    """
    # Get current position of the node
    current_position = node.translation if node.translation is not None else np.zeros(3)

    # 1. Apply translation
    translated_position = current_position + perturbation.translation

    # 2. Apply rotation around specified point
    # First get position relative to rotation point
    relative_pos = translated_position - perturbation.rotation_point

    # Rotate the relative position
    rotated_pos = quaternion_rotate(relative_pos, perturbation.rotation)

    # Transform back to world coordinates
    new_position = rotated_pos + perturbation.rotation_point

    # Update node position
    node.translation = new_position

    # Update node rotation
    if node.rotation is None:
        node.rotation = perturbation.rotation
    else:
        node.rotation = quaternion_multiply(perturbation.rotation, node.rotation)

# The original function can now be implemented as:
def granular_perturbation(node: pyrender.Node, config: PerturbationConfig) -> None:
    """
    Apply controlled random translation and rotation to the pyrender node.

    :param node: The pyrender Node to perturb
    :param config: PerturbationConfig object containing the ranges and rotation axis
    """
    perturbation = sample_perturbation(config)
    apply_perturbation(node, perturbation)


@dataclass
class SceneConfig:
    viewport_width: int = 1280
    viewport_height: int = 720
    bg_color: Tuple = (0.0, 0.0, 0.0, 0.0)
    light_intensity: float = 4.0
    light_color: Tuple = (1.0, 1.0, 1.0)
    camera_fov: float = np.pi / 5.0


class SceneHandler:
    def __init__(self, config: SceneConfig = SceneConfig()):
        self.config = config
        self.scene = pyrender.Scene(bg_color=config.bg_color)
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=config.viewport_width, viewport_height=config.viewport_height
        )

        self.cov_mesh_node: Optional[pyrender.Node] = None
        self.rad_mesh_node: Optional[pyrender.Node] = None
        self.camera_node: Optional[pyrender.Node] = None
        self.light_node: Optional[pyrender.Node] = None

        self.camera = pyrender.PerspectiveCamera(yfov=config.camera_fov)
        self.light = pyrender.DirectionalLight(
            color=config.light_color, intensity=config.light_intensity
        )

    @classmethod
    def from_stl_files(
        cls, mesh_path: str, rad_mesh_path: str, config: SceneConfig = SceneConfig()
    ) -> "SceneHandler":
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

    def _preprocess_meshes(
        self, cov_mesh: trimesh.Trimesh, rad_mesh: trimesh.Trimesh
    ) -> None:
        """Apply transformations to meshes"""

        # Scale to unit box
        bounds = cov_mesh.bounds
        scale_factor = 1.0 / np.max(bounds[1] - bounds[0])

        cov_mesh.apply_translation(COV_ORIGIN_OFFSET)
        cov_mesh.apply_scale(scale_factor)

        rad_mesh.apply_translation(RAD_ORIGIN_OFFSET)
        rad_mesh.apply_scale(scale_factor)

    def _create_pyrender_meshes(
        self, cov_mesh: trimesh.Trimesh, rad_mesh: trimesh.Trimesh
    ) -> None:
        """Create pyrender meshes with materials"""
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1, 1, 1, 1.0],
            metallicFactor=0.1,
            roughnessFactor=0.1,
            smooth=False,
            alphaMode="OPAQUE",
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

    def apply_perturbation(
        self,
        pertub: Perturbation,
    ) -> None:
        """
        Apply random perturbation to the meshes using granular control configuration.

        :param config: PerturbationConfig object containing perturbation ranges.
                    If None, uses default configuration.
        """
        if self.cov_mesh_node:
            apply_perturbation(self.cov_mesh_node, pertub)
        else:
            print("[ScenerHandler] No cover mesh node found to apply perturbation.")

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
    config = SceneConfig(viewport_width=640, viewport_height=480, light_intensity=4.0)
    handler = SceneHandler.from_stl_files(
        mesh_path=COUVERCLE_PATH, rad_mesh_path=BOITIER_PATH, config=config
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
