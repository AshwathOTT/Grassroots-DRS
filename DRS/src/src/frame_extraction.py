import cv2
import os

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read and save each frame
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:  # If no more frames, break
            break
        cv2.imwrite(f"{output_folder}/frame{frame_number:04d}.jpg", frame)  # Save frame
        frame_number += 1

    cap.release()
    print(f"Extracted {frame_number} frames to '{output_folder}'.")

# Test the function
if __name__ == "__main__":
    # Update the file paths to match your project structure
    video_file = "/Users/ashwath.s49/PycharmProjects/grassRootsCricket_suites/DRS/data/sample.mp4"
    # Path to the video inside the DRS project
    output_dir = "frames"  # Output folder in the root of the DRS project
    extract_frames(video_file, output_dir)
