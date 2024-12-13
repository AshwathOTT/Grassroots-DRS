import os
import cv2
import numpy as np
from scipy.interpolate import make_interp_spline

def detect_stumps(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    lower_pink = np.array([135, 40, 40])
    upper_pink = np.array([175, 255, 255])
    stumps_mask = cv2.inRange(hsv, lower_pink, upper_pink)

    stumps_contours, _ = cv2.findContours(stumps_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if stumps_contours:
        x_min, y_min, x_max, y_max = float("inf"), float("inf"), 0, 0
        for contour in stumps_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min, y_min, x_max, y_max = min(x_min, x), min(y_min, y), max(x_max, x + w), max(y_max, y + h)

        frame_height, frame_width = frame.shape[:2]
        y_min, y_max = max(0, y_min - 56), min(frame_height, y_max + 95)
        x_min, x_max = max(0, x_min - 3), min(frame_width, x_max + 39)

        return x_min, y_min, x_max - x_min, y_max - y_min
    else:
        print("Error: Stumps not detected.")
        return None


def calculate_initial_velocity(ball_positions, frame_rate):
    if len(ball_positions) < 2:
        print("Not enough ball positions to calculate velocity.")
        return None

    x1, y1 = ball_positions[-2]
    x2, y2 = ball_positions[-1]

    time_interval = 1 / frame_rate
    vx = (x2 - x1) / time_interval
    vy = (y2 - y1) / time_interval

    return vx, vy


def predict_trajectory_with_visualization(frame, ball_position, initial_velocity, stumps_position, frame_rate):
    if initial_velocity is None:
        print("No valid initial velocity provided.")
        return frame, False

    vx, vy = initial_velocity
    x, y = ball_position
    gravity = 9.8  # Gravity in m/s^2

    stumps_x, stumps_y, stumps_width, stumps_height = stumps_position
    stumps_x_end, stumps_y_end = stumps_x + stumps_width, stumps_y + stumps_height

    hit_stumps = False
    collision_point = None
    trajectory_points = []

    for t in np.arange(0, 2, 1 / frame_rate):
        x_t = x + vx * t
        y_t = y + vy * t - 0.5 * gravity * (t ** 2)  # Parabolic motion

        if x_t < 0 or x_t >= frame.shape[1] or y_t < 0 or y_t >= frame.shape[0]:
            break

        trajectory_points.append((x_t, y_t))

        if stumps_x <= x_t <= stumps_x_end and stumps_y <= y_t <= stumps_y_end:
            collision_point = (int(x_t), int(y_t))
            hit_stumps = True
            break

    # Smooth trajectory rendering
    if len(trajectory_points) >= 4:
        trajectory_points = np.array(trajectory_points)
        x_coords, y_coords = trajectory_points[:, 0], trajectory_points[:, 1]

        t_smooth = np.linspace(0, len(x_coords) - 1, 300)
        spline_x = make_interp_spline(np.arange(len(x_coords)), x_coords, k=3)(t_smooth)
        spline_y = make_interp_spline(np.arange(len(y_coords)), y_coords, k=3)(t_smooth)

        for i in range(1, len(spline_x)):
            pt1 = (int(spline_x[i - 1]), int(spline_y[i - 1]))
            pt2 = (int(spline_x[i]), int(spline_y[i]))
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)  # Red line for trajectory
    else:
        for i in range(1, len(trajectory_points)):
            pt1 = (int(trajectory_points[i - 1][0]), int(trajectory_points[i - 1][1]))
            pt2 = (int(trajectory_points[i][0]), int(trajectory_points[i][1]))
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

    if collision_point:
        cv2.circle(frame, collision_point, 10, (0, 255, 0), -1)
        cv2.putText(frame, "Collision Point", (collision_point[0] - 30, collision_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.rectangle(frame, (stumps_x, stumps_y), (stumps_x_end, stumps_y_end), (255, 0, 255), 2)

    return frame, hit_stumps


if __name__ == "__main__":
    video_path = "/Users/ashwath.s49/PycharmProjects/grassRootsCricket_suites/DRS/data/sample.mp4"
    output_folder = "output_frames"
    os.makedirs(output_folder, exist_ok=True)
    ball_positions = [(400, 500), (410, 490), (420, 480)]  # Sample positions
    frame_rate = 30

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    stumps_position = None
    initial_velocity = calculate_initial_velocity(ball_positions, frame_rate)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == 0:
            stumps_position = detect_stumps(frame)
            if stumps_position is None:
                print("Stumps not detected.")
                break

        if frame_idx < len(ball_positions):
            ball_position = ball_positions[frame_idx]
        else:
            ball_position = ball_positions[-1]

        frame, hit_stumps = predict_trajectory_with_visualization(
            frame, ball_position, initial_velocity, stumps_position, frame_rate
        )

        output_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, frame)

        frame_idx += 1
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    print(f"Frames saved to {output_folder}")
