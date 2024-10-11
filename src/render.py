from typing import Tuple
import numpy as np
import pyrender
import trimesh


def preprocess_mesh(path: str) -> pyrender.Mesh:
    mesh = trimesh.load(path)
    mesh.apply_translation(-mesh.centroid)

    # Scale the model to fit in a unit box
    bounds = mesh.bounds
    scale_factor = 1.0 / np.max(bounds[1] - bounds[0])
    mesh.apply_scale(scale_factor)

    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.vertex_normals = mesh.vertex_normals

    mesh = trimesh.graph.smooth_shade(mesh)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1, 1, 1, 1.0],
        metallicFactor=0.1,
        roughnessFactor=0.1,
        smooth=False,
        alphaMode="OPAQUE",
    )
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    return mesh


def init_cameras() -> Tuple[pyrender.PerspectiveCamera, Tuple[np.ndarray]]:
    # Define camera parameters
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    side_camera_pose = np.array(
        [
            [0.0, 0.0, 1.0, 2.0],
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


def render(path: str) -> None:
    mesh = preprocess_mesh(path)

    scene = pyrender.Scene(bg_color=np.zeros(4))  # , ambient_light=np.ones(3))
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

    scene.add(mesh)
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

    imgs = render("./assets/3dmodels/couvercle.stl")

    img_names = ["top_view.png", "side_view.png"]
    for img, name in zip(imgs, img_names):
        Image.fromarray(img).save(name)
