import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

import cv2
from preprocessing import OpenCVImage, create_open_cv_image
from detection import RadiatorHandler, CoverHandler
from render import (
    Perturbation,
    PerturbationConfig,
    SceneHandler,
    RotationAxis,
    COUVERCLE_PATH,
    BOITIER_PATH,
    OUTPUT_PATH,
    TOP_CAMERA_POSE,
    SIDE_CAMERA_POSE,
    TOP_LIGHT_POSE,
    SIDE_LIGHT_POSE,
    sample_perturbation,
    quaternion_multiply,
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


def calculate_perturbation_error(initial: Perturbation, estimated_correction: Perturbation) -> float:
    """
    Calculate relative error between estimated correction and theoretical correction (reverse of initial)

    Args:
        initial: Initial perturbation applied
        estimated_correction: Estimated correction perturbation

    Returns:
        float: Relative error between 0 and 1, where 0 means perfect correction
    """
    # Theoretical correction should be the reverse of initial perturbation
    theoretical_correction = Perturbation(
        translation=-initial.translation,
        rotation=np.array([*-initial.rotation[:3], initial.rotation[3]]),  # Inverse quaternion
        rotation_point=initial.rotation_point
    )

    # Calculate translation error (Euclidean distance)
    translation_error = np.linalg.norm(
        estimated_correction.translation - theoretical_correction.translation
    )

    # Calculate rotation error (quaternion distance)
    # Using dot product between quaternions: error = 1 - |q1·q2|
    rotation_error = 1 - abs(np.dot(estimated_correction.rotation,
                                  theoretical_correction.rotation))

    # Calculate rotation point error
    rotation_point_error = np.linalg.norm(
        estimated_correction.rotation_point - theoretical_correction.rotation_point
    )

    # Combine errors with weights
    w1, w2, w3 = 0.5, 0.5, 0.0  # Weights for translation, rotation, and rotation point
    max_translation_error = 1.0  # Expected maximum translation error in units
    max_rotation_point_error = 1.0  # Expected maximum rotation point error in units

    relative_error = (
        w1 * min(translation_error / max_translation_error, 1.0) +
        w2 * rotation_error +
        w3 * min(rotation_point_error / max_rotation_point_error, 1.0)
    )

    return min(max(relative_error, 0.0), 1.0)


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
) -> float:
    """Simulates robotic movements and records a video of hole position estimation."""
    scene = SceneHandler.from_stl_files(COUVERCLE_PATH, BOITIER_PATH)
    init_pertub = sample_perturbation(PerturbationConfig())
    scene.apply_perturbation(init_pertub)
    correction_error = -1.0

    # Video file setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps=1, frameSize=(1280, 720))
    for step in range(1, num_steps + 1):

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
        x_translation = np.array(rad_handler._hole)[1] - np.array(cover_handler._right_hole)[1]

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
        correction_error = calculate_perturbation_error(init_pertub, correction)
        print(f"Initial Pertubation: {init_pertub}")
        print(f"Correction Matrix: {correction}")
        print(f"Relative Error: {correction_error:.4f}")

        scene.apply_perturbation(correction)
        rad_img, cover_img_top, cover_img_side = render_all(scene)

        # video_writer.write(cercle_result.image)

        # Display simulation in real time
        cv2.imshow("Result of Correction Side", cover_img_side.cv_image)
        cv2.imshow("Result of Correction Top", cover_img_top.cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(f"{OUTPUT_PATH}/ResultSide.png", cover_img_side.cv_image)
        cv2.imwrite(f"{OUTPUT_PATH}/ResultTop.png", cover_img_top.cv_image)

        # if cv2.waitKey(10) & 0xFF == ord("q"):
        #     break

        # time.sleep(0.2)

    # Post-simulation cleaning
    scene.cleanup()
    video_writer.release()
    cv2.destroyAllWindows()

    return correction_error

def simulate_correction_convergence(num_steps=10) -> None:
    """
    Simulates multiple correction steps and plots the error evolution.
    Records translation and rotation errors at each step.
    """
    scene = SceneHandler.from_stl_files(COUVERCLE_PATH, BOITIER_PATH)

    # Initialize arrays to store errors
    translation_errors = []
    rotation_errors = []
    correction_errors = []
    steps = []

    # Apply initial perturbation
    init_pertub = sample_perturbation(PerturbationConfig())
    scene.apply_perturbation(init_pertub)
    print(f"Initial Perturbation:\n{init_pertub}")

    # Initialize cumulative correction
    cumulative_correction = Perturbation(
        translation=np.zeros(3),
        rotation=np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
        rotation_point=np.zeros(3)
    )

    for step in range(1, num_steps + 1):
        print(f"\nStep {step}/{num_steps}:")

        # Render views and process images
        rad_img, cover_img_top, cover_img_side = render_all(scene)

        # Process images
        rad_handler = RadiatorHandler(rad_img)
        rad_handler.process_image()

        cover_handler = CoverHandler(cover_img_top, cover_img_side)
        cover_handler.process_image()

        # Get measurements
        roll = cover_handler.get_roll()
        x_translation = np.array(rad_handler._hole)[1] - np.array(cover_handler._right_hole)[1]

        # Compute correction for this step
        step_correction = compute_simplified_correction(
            x_translation, roll, RotationAxis()
        )
        print(f"Step Correction: {step_correction}")
        print(f"Translation Error: {x_translation:.2f} px")
        print(f"Rotation Error: {roll:.2f} deg")

        # Update cumulative correction
        # Combine translations
        cumulative_correction.translation += step_correction.translation

        # Combine rotations using quaternion multiplication
        cumulative_correction.rotation = quaternion_multiply(
            step_correction.rotation,
            cumulative_correction.rotation
        )

        # The rotation point remains the same as we're using the same axis

        # Calculate and store errors
        correction_error = calculate_perturbation_error(init_pertub, cumulative_correction)
        translation_errors.append(abs(x_translation))
        rotation_errors.append(abs(roll))
        correction_errors.append(correction_error)
        steps.append(step)

        print(f"Cumulative Correction: {cumulative_correction}")
        print(f"Cumulative Correction Error: {correction_error:.4f}")

        # Apply step correction
        scene.apply_perturbation(step_correction)

    rad_img, cover_img_top, cover_img_side = render_all(scene)
    cv2.imwrite(f"{OUTPUT_PATH}/ResultSide.png", cover_img_side.cv_image)
    cv2.imwrite(f"{OUTPUT_PATH}/ResultTop.png", cover_img_top.cv_image)

    # Create plot with adjusted figure size for LaTeX document
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # Plot translation error
    ax1.plot(steps, translation_errors, color='black', markersize=4)
    ax1.set_title('Erreur de translation vs. Itération', pad=10)
    ax1.set_xlabel('Itération')
    ax1.set_ylabel('Erreur de translation (pixels)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot rotation error
    ax2.plot(steps, rotation_errors, color='black', markersize=4)
    ax2.set_title('Erreur de rotation vs. Itération', pad=10)
    ax2.set_xlabel('Itération')
    ax2.set_ylabel('Erreur de rotation (degrees)')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot correction error
    ax3.plot(steps, correction_errors, color='black', markersize=4)
    ax3.set_title('Erreur relatif vs Perturbation initiale', pad=10)
    ax3.set_xlabel('Itération')
    ax3.set_ylabel('Erreur de correction')
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout with tighter spacing
    plt.tight_layout(pad=1.5)

    # Save with higher DPI for better quality
    plt.savefig(f"{OUTPUT_PATH}/convergence_plot.pdf",
                format='pdf',
                dpi=300,
                bbox_inches='tight')
    plt.savefig(f"{OUTPUT_PATH}/convergence_plot.png",
                dpi=300,
                bbox_inches='tight')
    plt.show()

    # Cleanup
    scene.cleanup()

if __name__ == "__main__":
    # Simulate robotic movements on the mesh, estimate hole positions and record video
    # _ = simulate_robotic_movement(
    #     num_steps=1,
    #     output_video=SIDE_SIMULATION_PATH,
    # )

    simulate_correction_convergence(num_steps=10)
