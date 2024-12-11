import time
import numpy as np
from typing import Tuple
import cv2
from preprocessing import OpenCVImage, create_open_cv_image
from detection import RadiatorHandler, CoverHandler
from render import (
    Perturbation,
    PerturbationConfig,
    SceneHandler,
    RotationAxis,
    apply_perturbation,
    COUVERCLE_PATH,
    BOITIER_PATH,
    OUTPUT_PATH,
    TOP_CAMERA_POSE,
    SIDE_CAMERA_POSE,
    TOP_LIGHT_POSE,
    SIDE_LIGHT_POSE,
    sample_perturbation,
)

TOP_SIMULATION_PATH = f"{OUTPUT_PATH}/robotic_top_simulation.mp4"
SIDE_SIMULATION_PATH = f"{OUTPUT_PATH}/robotic_side_simulation.mp4"


def compute_simplified_correction(
    delta_x_pixels,
    roll_angle_degrees,
    rotation_axis: RotationAxis,
    fov=np.pi/5.0,
    image_width=1280,
    distance=2.0
) -> Perturbation:
    """
    Computes a correction as a Perturbation object that undoes a translation along X and a rotation around an arbitrary axis.

    Parameters:
    - delta_x_pixels: Translation in pixels along the X-axis (detected from the TOP camera).
    - roll_angle_degrees: Roll angle in degrees (detected from the FRONT camera).
    - rotation_axis: RotationAxis object defining the axis of rotation.
    - fov: Field of view in radians (default np.pi/5).
    - image_width: Image width in pixels (default 1280).
    - distance: Distance from the camera to the ground plane in world units (default 2.0).

    Returns:
    - Perturbation object containing the inverse transformation
    """
    # Step 1: Calculate GSD (Ground Sample Distance) for the TOP camera
    gsd = (2 * distance * np.tan(fov / 2)) / image_width

    # Step 2: Convert pixel translation to world units and create translation vector
    # Use positive delta_x_world to reverse the detected negative translation
    delta_x_world = delta_x_pixels * gsd
    translation = np.array([-delta_x_world, 0.0, 0.0])

    # Step 3: Convert roll angle from degrees to radians
    # Negate the angle to reverse the rotation
    roll_angle_radians = np.deg2rad(-roll_angle_degrees)  # Added negative sign

    # Step 4: Create rotation quaternion around arbitrary axis
    # First normalize the direction vector
    direction = rotation_axis.direction / np.linalg.norm(rotation_axis.direction)

    # Convert to quaternion (using negated angle)
    half_angle = roll_angle_radians / 2.0
    qx = direction[0] * np.sin(half_angle)
    qy = direction[1] * np.sin(half_angle)
    qz = direction[2] * np.sin(half_angle)
    qw = np.cos(half_angle)

    # Create and normalize quaternion
    quaternion = np.array([qx, qy, qz, qw])
    quaternion /= np.linalg.norm(quaternion)

    return Perturbation(
        translation=translation,
        rotation=quaternion,
        rotation_point=rotation_axis.point
    )


def render_all(scene: SceneHandler) -> Tuple[OpenCVImage, OpenCVImage, OpenCVImage] :
    # Top View rendering
    scene.set_camera_pose(TOP_CAMERA_POSE)
    scene.set_light_pose(TOP_LIGHT_POSE)

    rad_img = scene.render(show_cov=False)
    rad_img = create_open_cv_image(rad_img)

    cover_img_top = scene.render(show_cov=True)

    # Side View rendering
    scene.set_camera_pose(SIDE_CAMERA_POSE)
    scene.set_light_pose(SIDE_LIGHT_POSE)

    cover_img_top = create_open_cv_image(cover_img_top)
    cover_img_side = scene.render(show_cov=True)
    cover_img_side = create_open_cv_image(cover_img_side)

    return rad_img, cover_img_top, cover_img_side

def simulate_robotic_movement(
    num_steps=50,
    output_video="simulation_robotic.mp4"
) -> None:
    """Simulates robotic movements and records a video of hole position estimation."""
    scene = SceneHandler.from_stl_files(COUVERCLE_PATH, BOITIER_PATH)
    init_pertub = sample_perturbation(PerturbationConfig())
    scene.apply_perturbation(init_pertub)

    # Video file setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps=1, frameSize=(1280, 720))

    for step in range(num_steps):

        # Render all views
        rad_img, cover_img_top, cover_img_side = render_all(scene)

        rad_handler = RadiatorHandler(rad_img)
        rad_handler.process_image()
        rad_handler.display_result()

        cover_handler = CoverHandler(cover_img_top, cover_img_side)
        cover_handler.process_image()
        cover_handler.display_result()

        # Compile results
        roll = cover_handler.get_roll()
        x_translation = np.array(rad_handler._hole)[0] - np.array(cover_handler._right_hole)[0]

        print(f"Results {step}/{num_steps}:")
        print(f"Radiator Hole: {rad_handler._hole}")
        print(f"Cover Hole: Left {cover_handler._left_hole} | Right {cover_handler._right_hole}")
        print(f"Cover Roll: {roll:.3f} deg.")
        print(f"Cover Translation (X): {x_translation} px.")

        # Compute correction matrix
        correction: Perturbation = compute_simplified_correction(
            x_translation,
            roll,
            RotationAxis()
        )
        print(f"Initial Pertubation: {init_pertub}")
        print(f"Correction Matrix: {correction}")

        scene.apply_perturbation(correction)
        rad_img, cover_img_top, cover_img_side = render_all(scene)

        # video_writer.write(cercle_result.image)

        # Display simulation in real time
        cv2.imshow("Result of Correction", cover_img_side.cv_image)
        cv2.imshow("Result of Correction", cover_img_top.cv_image)
        cv2.waitKey(0)

        cv2.imwrite(f"{OUTPUT_PATH}/ResultSide.png", cover_img_side.cv_image)
        cv2.imwrite(f"{OUTPUT_PATH}/ResultTop.png", cover_img_top.cv_image)

        # if cv2.waitKey(10) & 0xFF == ord("q"):
        #     break

        # time.sleep(0.2)

    # Post-simulation cleaning
    scene.cleanup()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Simulate robotic movements on the mesh, estimate hole positions and record video
    simulate_robotic_movement(
        num_steps=1,
        output_video=SIDE_SIMULATION_PATH,
    )
