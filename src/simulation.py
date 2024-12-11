import time
import numpy as np
import cv2
from preprocessing import create_open_cv_image
from detection import RadiatorHandler, CoverHandler
from render import (
    SceneHandler,
    COUVERCLE_PATH,
    BOITIER_PATH,
    OUTPUT_PATH,
    TOP_CAMERA_POSE,
    SIDE_CAMERA_POSE,
    TOP_LIGHT_POSE,
    SIDE_LIGHT_POSE,
)

TOP_SIMULATION_PATH = f"{OUTPUT_PATH}/robotic_top_simulation.mp4"
SIDE_SIMULATION_PATH = f"{OUTPUT_PATH}/robotic_side_simulation.mp4"


def compute_simplified_correction(
    delta_x_pixels,
    roll_angle_degrees,
    fov=np.pi/5.0,
    image_width=1280,
    distance=2.0
):
    """
    Computes a transformation matrix for a translation along -X and a rotation around +X (roll).

    Parameters:
    - delta_x_pixels: Translation in pixels along the X-axis (detected from the TOP camera).
    - roll_angle_degrees: Roll angle in degrees (detected from the FRONT camera).
    - fov: Field of view in radians (default np.pi/5).
    - image_width: Image width in pixels (default 1280).
    - distance: Distance from the camera to the ground plane in world units (default 2.0).

    Returns:
    - transformation_matrix: 4x4 numpy array representing the correction transformation.
    """

    # Step 1: Calculate GSD (Ground Sample Distance) for the TOP camera
    gsd = (2 * distance * np.tan(fov / 2)) / image_width

    # Step 2: Convert pixel translation to world units
    delta_x_world = delta_x_pixels * gsd

    # Step 3: Convert roll angle from degrees to radians
    roll_angle_radians = np.deg2rad(roll_angle_degrees)

    # Step 4: Construct the transformation matrix
    # Translation along -X and rotation around +X
    translation = np.array([
        [1, 0, 0, -delta_x_world],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    rotation_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll_angle_radians), -np.sin(roll_angle_radians), 0],
        [0, np.sin(roll_angle_radians), np.cos(roll_angle_radians), 0],
        [0, 0, 0, 1]
    ])

    # Combined transformation: Translation followed by rotation
    transformation_matrix = rotation_x @ translation

    return transformation_matrix


def simulate_robotic_movement(
    num_steps=50,
    output_video="simulation_robotic.mp4"
) -> None:
    """Simulates robotic movements and records a video of hole position estimation."""
    scene = SceneHandler.from_stl_files(COUVERCLE_PATH, BOITIER_PATH)
    scene.apply_perturbation()

    # Video file setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps=1, frameSize=(1280, 720))

    for step in range(num_steps):

        # Top View rendering
        scene.set_camera_pose(TOP_CAMERA_POSE)
        scene.set_light_pose(TOP_LIGHT_POSE)

        rad_img = scene.render(show_cov=False)
        rad_img = create_open_cv_image(rad_img)

        rad_handler = RadiatorHandler(rad_img)
        rad_handler.process_image()
        rad_handler.display_result()

        cover_img_top = scene.render(show_cov=True)

        # Side View rendering
        scene.set_camera_pose(SIDE_CAMERA_POSE)
        scene.set_light_pose(SIDE_LIGHT_POSE)

        cover_img_top = create_open_cv_image(cover_img_top)
        cover_img_side = scene.render(show_cov=True)
        cover_img_side = create_open_cv_image(cover_img_side)

        cover_handler = CoverHandler(cover_img_top, cover_img_side)
        cover_handler.process_image()
        cover_handler.display_result()

        print(f"Results {step}/{num_steps}:")
        print(f"Radiator Hole: {rad_handler._hole}")
        print(f"Cover Hole: Left {cover_handler._left_hole} | Right {cover_handler._right_hole}")
        print(f"Cover Roll: {cover_handler.get_roll():.3f} deg.")
        print(f"Cover Translation (X, Z): {np.array(rad_handler._hole)[:2] - np.array(cover_handler._right_hole)[:2]} px.")

        # video_writer.write(cercle_result.image)

        # Display simulation in real time
        # cv2.imshow("Simulated Robotic Movement", cercle_result.image)
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
