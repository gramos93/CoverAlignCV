import time

import cv2
import numpy as np
import pyrender

from detection import (
    OpenCVImage,
    detect_cercles,
)
from render import (
    init_cameras,
    init_lights,
    random_perturbation,
    preprocess_trimesh
)


def simulate_robotic_movement(
    file_path: str,
    num_steps=50,
    output_video="simulation_robotic.mp4"
) -> None:
    """Simulates robotic movements and records a video of hole position estimation."""
    # Preprocess mesh
    mesh = preprocess_trimesh(file_path)

    # Create a scene, renderer and material
    scene = pyrender.Scene(bg_color=np.zeros(4))
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1, 1, 1, 1.0],
        metallicFactor=0.1,
        roughnessFactor=0.1,
        smooth=False,
        alphaMode="OPAQUE",
    )

    # Add light on scene
    light, light_poses = init_lights()
    scene.add(light, pose=light_poses[0])

    # Add camera on scene
    camera, cam_poses = init_cameras()
    scene.add(camera, pose=cam_poses[0])

    # Video file setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps=1, frameSize=(640, 480))

    # Simulation of robotic movements
    for step in range(num_steps):
        # Apply random perturbation to simulate robotic imprecision
        random_perturbation(mesh)

        # Add mesh on scene
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node = scene.add(pyrender_mesh)

        # Write the scene image into a video
        color, _ = renderer.render(scene)
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        processed_image = OpenCVImage(
            np_image=color,
            cv_image=color_bgr,
            gray=cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        )
        circle_result = detect_cercles(processed_image)
        video_writer.write(circle_result.image)

        # Display simulation in real time
        cv2.imshow('Simulated Robotic Movement', circle_result.image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        time.sleep(0.2)
        scene.remove_node(mesh_node)

    # Post-simulation cleaning
    renderer.delete()
    video_writer.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Simulate robotic movements on the mesh, estimate hole positions and record video
    simulate_robotic_movement(
        file_path="../assets/3dmodels/couvercle.stl",
        num_steps=100,
        output_video="robotic_simulation.mp4"
    )

