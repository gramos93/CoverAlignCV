from typing import Tuple
import numpy as np
import pyrender
import trimesh

COUVERCLE_PATH = "./assets/3dmodels/couvercle.stl"
BOITIER_PATH = "./assets/3dmodels/boitier.stl"
ORIGIN_OFFSET = np.array([651.86, 573.76, -2_894.40])


def random_perturbation(
    mesh: trimesh.Trimesh,
    translation_range: float = 0.02,
    rotation_range: float = np.pi / 36,
) -> None:
    """
    Apply random translation and rotation to the mesh to simulate robotic imprecision.

    :param mesh: The 3D mesh to perturb.
    :param translation_range: Max translation perturbation (in normalized units).
    :param rotation_range: Max rotation perturbation (in radians).
    """
    # Random translation within the given range
    translation = np.random.uniform(-translation_range, translation_range, size=3)
    mesh.apply_translation(translation)

    # Random rotation about the X, Y, Z axes
    rotation_axis = np.random.normal(size=3)
    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize to make it a unit vector
    rotation_angle = np.random.uniform(-rotation_range, rotation_range)
    rotation_matrix = trimesh.transformations.rotation_matrix(
        rotation_angle, rotation_axis
    )

    mesh.apply_transform(rotation_matrix)


def preprocess_meshes() -> pyrender.Mesh:
    mesh, rad_mesh = preprocess_trimesh()

    # Apply random perturbation to simulate robotic imprecision
    random_perturbation(mesh)

    # Ensure smooth shading and create a mesh with a flat material
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1, 1, 1, 1.0],
        metallicFactor=0.1,
        roughnessFactor=0.1,
        smooth=False,
        alphaMode="OPAQUE",
    )
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    pyrender_rad_mesh = pyrender.Mesh.from_trimesh(rad_mesh, material=material)

    return pyrender_mesh, pyrender_rad_mesh


def preprocess_trimesh() -> Tuple[trimesh.Trimesh]:
    mesh = trimesh.load(COUVERCLE_PATH)
    rad_mesh = trimesh.load(BOITIER_PATH)

    # Scale the model to fit in a unit box
    bounds = mesh.bounds
    scale_factor = 1.0 / np.max(bounds[1] - bounds[0])

    mesh.apply_translation(ORIGIN_OFFSET)
    rad_mesh.apply_translation(ORIGIN_OFFSET)

    mesh.apply_scale(scale_factor)
    rad_mesh.apply_scale(scale_factor)

    return mesh, rad_mesh


def init_cameras() -> Tuple[pyrender.PerspectiveCamera, Tuple[np.ndarray]]:
    """
    Camera frame of reference guide
    https://pyrender.readthedocs.io/en/latest/examples/cameras.html
    """
    # Define camera parameters
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    side_camera_pose = np.array(
        [
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    top_camera_pose = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    return camera, (top_camera_pose, side_camera_pose)


def init_lights() -> Tuple[pyrender.DirectionalLight, Tuple[np.ndarray]]:
    top_light_pose = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 10.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    side_light_pose = np.array(
        [
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    return directional_light, (top_light_pose, side_light_pose)


def render() -> list[np.ndarray]:
    mesh, rad_mesh = preprocess_meshes()

    scene = pyrender.Scene(bg_color=np.zeros(4))  # , ambient_light=np.ones(3))
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

    scene.add(mesh)
    scene.add(rad_mesh)
    camera, cam_poses = init_cameras()
    light, light_poses = init_lights()

    renders = []
    for cam_pose, light_pose in zip(cam_poses, light_poses):
        scene.add(light, pose=light_pose)
        cam_node = scene.add(camera, pose=cam_pose)
        color_side, _ = renderer.render(scene)
        renders.append(color_side)
        scene.remove_node(cam_node)

    renderer.delete()
    return renders


if __name__ == "__main__":
    from PIL import Image

    imgs = render()
    img_names = ["top_view.png", "side_view.png"]
    for img, name in zip(imgs, img_names):
        Image.fromarray(img).save(name)
