import time
import cv2

from detection import (
    OpenCVImage,
    detect_cercles,
    detect_cercles_in_cover_area,
)
from render import (
    SceneHandler,
    COUVERCLE_PATH,
    BOITIER_PATH,
    TOP_CAMERA_POSE,
    SIDE_CAMERA_POSE,
    TOP_LIGHT_POSE,
    SIDE_LIGHT_POSE,
)

SIMULATION_PATH = r"./outputs/robotic_simulation.mp4"



def simulate_robotic_movement(
    num_steps=50, output_video="simulation_robotic.mp4"
) -> None:
    """Simulates robotic movements and records a video of hole position estimation."""
    scene = SceneHandler.from_stl_files(COUVERCLE_PATH, BOITIER_PATH)
    scene.set_camera_pose(TOP_CAMERA_POSE)
    scene.set_light_pose(TOP_LIGHT_POSE)

    # Video file setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps=1, frameSize=(640, 480))

    # Simulation of robotic movements
    for step in range(num_steps):
        # Apply random perturbation to simulate robotic imprecision
        scene.apply_random_perturbation()
        # This line allow the addition of the cover at some point in the loop.
        # In this case after step 10.
        color = scene.render(show_cov=True if step > 10 else False)

        # Write the scene image into a video
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        processed_image = OpenCVImage(
            np_image=color,
            cv_image=color_bgr,
            gray=cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY),
        )
        cercle_result = detect_cercles_in_cover_area(processed_image)
        video_writer.write(cercle_result.image)

        # Display simulation in real time
        cv2.imshow("Simulated Robotic Movement", cercle_result.image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

        time.sleep(0.2)

    # Post-simulation cleaning
    scene.cleanup()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Simulate robotic movements on the mesh, estimate hole positions and record video
    simulate_robotic_movement(
        num_steps=100,
        output_video=SIMULATION_PATH,
    )
