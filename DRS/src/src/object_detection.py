import cv2
import numpy as np
import os

# Kalman filter for ball tracking
def initialize_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)  # Corrected shape
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman


# Function to detect stumps in the first frame
def detect_stumps_in_first_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    lower_pink = np.array([135, 50, 50])
    upper_pink = np.array([175, 255, 255])
    stumps_mask = cv2.inRange(hsv, lower_pink, upper_pink)

    stumps_contours, _ = cv2.findContours(stumps_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if stumps_contours:
        x_min = float("inf")
        y_min = float("inf")
        x_max = 0
        y_max = 0
        for contour in stumps_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        frame_height, frame_width = frame.shape[:2]
        y_min = max(0, y_min - 56)
        y_max = min(frame_height, y_max + 95)
        x_min = max(0, x_min - 3)
        x_max = min(frame_width, x_max + 39)

        stumps_position = (x_min, y_min, x_max - x_min, y_max - y_min)
        return stumps_position
    else:
        print("Error: Stumps not detected in the first frame.")
        return None

# Detect ball anywhere in the frame
def detect_ball_in_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

    # Detect red regions
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    ball_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    ball_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    ball_mask = cv2.bitwise_or(ball_mask1, ball_mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)

    ball_contours, _ = cv2.findContours(ball_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ball_position = None
    if ball_contours:
        largest_contour = max(ball_contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        ball_position = (x + w // 2, y + h // 2)
        return ball_position
    return None

# Main script
if __name__ == "__main__":
    frame_folder = "frames"
    ball_positions = []
    stumps_position = None
    kalman = initialize_kalman_filter()

    for frame_idx, frame_file in enumerate(sorted(os.listdir(frame_folder))):
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is not None:
            if frame_idx == 0:
                stumps_position = detect_stumps_in_first_frame(frame)
                if stumps_position is None:
                    break
            else:
                # Detect ball
                ball_pos = detect_ball_in_frame(frame)
                if ball_pos:
                    ball_positions.append(ball_pos)
                    print(f"Ball detected at {ball_pos} in frame {frame_file}")

                    # Update Kalman filter
                    kalman.correct(np.array([[np.float32(ball_pos[0])], [np.float32(ball_pos[1])]]))
                else:
                    # Use Kalman filter prediction
                    predicted = kalman.predict()
                    ball_pos = (int(predicted[0]), int(predicted[1]))
                    print(f"Predicted ball position: {ball_pos}")
                    ball_positions.append(ball_pos)

                # Visualization
                if ball_pos:
                    cv2.circle(frame, ball_pos, 10, (0, 0, 255), -1)  # Red dot for ball
                if stumps_position:
                    x, y, w, h = stumps_position
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Pink box for stumps

                # Draw trajectory of the ball
                for i in range(1, len(ball_positions)):
                    pt1 = ball_positions[i - 1]
                    pt2 = ball_positions[i]
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green line for trajectory

                cv2.imshow("Detected Objects", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
        else:
            print(f"Error: Could not load frame {frame_file}")

    cv2.destroyAllWindows()
    print("\nTracked Ball Positions Across Frames:")
    for idx, pos in enumerate(ball_positions):
        print(f"Frame {idx}: {pos}")
