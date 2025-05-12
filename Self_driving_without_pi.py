import cv2
import numpy as np
import os
import glob
import time
from sklearn.cluster import KMeans

# Keep track of lane and object detection history
cluster_history = []
left_history = []
right_history = []
missing_lane_counter = 0

USE_KMEANS_MASK = True  # Use KMeans for object detection (set False for simple thresholding)

# Smooths lane or object detection over time
def average_history(history, new_value, max_len=5):
    if new_value is not None:
        history.append(new_value)
    if len(history) > max_len:
        history.pop(0)
    if history:
        x1s, y1s, x2s, y2s = zip(*history)
        return int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))
    return None

# Smooths object cluster centers using history
def smooth_kmeans_labels(current_labels, current_centroids, history, max_history=5):
    history.append(current_centroids)
    if len(history) > max_history:
        history.pop(0)
    avg_centroids = np.mean(history, axis=0)
    best_label = np.argmin(avg_centroids)
    return (current_labels == best_label).astype(np.uint8) * 255

# Averages similar lines into one
def average_line(lines, img_height, img_width):
    if len(lines) == 0:
        return None
    x_coords = []
    y_coords = []
    for x1, y1, x2, y2 in lines:
        x_coords += [x1, x2]
        y_coords += [y1, y2]
    poly = np.polyfit(y_coords, x_coords, 1)
    y1 = img_height
    y2 = int(img_height * 0.1)
    x1 = int(np.polyval(poly, y1))
    x2 = int(np.polyval(poly, y2))
    x1 = np.clip(x1, 0, img_width)
    x2 = np.clip(x2, 0, img_width)
    return x1, y1, x2, y2

# Main processing per video frame
def process_frame(frame, prev_time, fps_list):
    global missing_lane_counter

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_channel = cv2.GaussianBlur(s_channel, (3, 3), 0)

    height, width = s_channel.shape

    # Define region of interest (ROI) for road
    roi_vertices = np.array([[
        (width * 0.00, height),
        (width * 0.25, height * 0.4),
        (width * 0.75, height * 0.4),
        (width * 1.00, height)
    ]], dtype=np.int32)

    mask = np.zeros_like(s_channel)
    cv2.fillPoly(mask, [roi_vertices], 255)
    roi = cv2.bitwise_and(s_channel, s_channel, mask=mask)
    x, y, w, h = cv2.boundingRect(roi_vertices)
    cropped_roi = roi[y:y + h, x:x + w]

    # Object detection using KMeans or threshold
    if USE_KMEANS_MASK:
        reshaped = cropped_roi.reshape((-1, 1))
        kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
        kmeans.fit(reshaped)
        centroids = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_.reshape(cropped_roi.shape)
        clustered_roi = smooth_kmeans_labels(labels, centroids, cluster_history)
    else:
        clustered_roi = (cropped_roi < 50).astype(np.uint8) * 255

    cv2.imshow("clustered", clustered_roi)

    # Detect lane lines
    edges = cv2.Canny(clustered_roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=50, maxLineGap=30)

    hough_visual = cv2.cvtColor(cropped_roi, cv2.COLOR_GRAY2BGR)
    green_carpet_mask = np.zeros_like(cropped_roi)

    left_lines = []
    right_lines = []

    # Separate lines into left and right
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5 or abs(slope) > 2:
                continue
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

    left_line = average_line(left_lines, h, w)
    right_line = average_line(right_lines, h, w)

    left_avg = average_history(left_history, left_line)
    right_avg = average_history(right_history, right_line)

    # Use fallback if only one line is visible
    if left_avg and not right_avg:
        right_avg = (w - left_avg[0], left_avg[1], w - left_avg[2], left_avg[3])
    elif right_avg and not left_avg:
        left_avg = (w - right_avg[0], right_avg[1], w - right_avg[2], right_avg[3])

    if left_avg:
        cv2.line(hough_visual, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), (0, 255, 0), 3)
    if right_avg:
        cv2.line(hough_visual, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), (0, 255, 0), 3)

    # Check if object is in the driving path
    object_detected = False
    if left_avg and right_avg:
        carpet_pts = np.array([
            [left_avg[0], left_avg[1]],
            [left_avg[2], left_avg[3]],
            [right_avg[2], right_avg[3]],
            [right_avg[0], right_avg[1]]
        ], dtype=np.int32)

        cv2.fillPoly(green_carpet_mask, [carpet_pts], 255)
        road_mask_bgr = cv2.merge([np.zeros_like(green_carpet_mask), green_carpet_mask, np.zeros_like(green_carpet_mask)])
        hough_visual = cv2.addWeighted(hough_visual, 1, road_mask_bgr, 0.5, 0)

        intersection = cv2.bitwise_and(clustered_roi, green_carpet_mask)
        contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
            bbox_area = w_box * h_box

            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity > 0.9 and area > 3000 and bbox_area > 5000 and h_box / w_box < 2.5:
                    print("Large object detected on path!")
                    object_detected = True
                    cv2.rectangle(hough_visual, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 0, 255), 2)
                    break

    # Decide driving direction
    direction = "Unknown"
    if object_detected:
        direction = "Stop"
    elif left_avg and right_avg:
        missing_lane_counter = 0
        lane_center = (left_avg[0] + right_avg[0]) // 2
        frame_center = w // 2
        offset = lane_center - frame_center
        if abs(offset) < 20:
            direction = "Straight"
        elif offset < -20:
            direction = "Left"
        else:
            direction = "Right"
    else:
        missing_lane_counter += 1
        if missing_lane_counter < 10 and not object_detected:
            direction = "Straight"
        else:
            direction = "Stop"

    cv2.putText(hough_visual, f"Direction: {direction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    print(f"Direction: {direction}")

    # Calculate and show FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    fps_list.append(fps)
    avg_fps = np.mean(fps_list[-10:])
    cv2.putText(cropped_roi, f"FPS: {avg_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show all windows
    frame_with_roi = frame.copy()
    cv2.polylines(frame_with_roi, [roi_vertices], True, (0, 255, 0), 2)
    cv2.imshow('1. Original with ROI', frame_with_roi)
    cv2.imshow('2. Saturation Channel ROI', cropped_roi)
    cv2.imshow('3. Hough Transform Lines', hough_visual)

    return current_time

# Runs the lane detection on a single video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    prev_time = time.time()
    fps_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        prev_time = process_frame(frame, prev_time, fps_list)

        key = cv2.waitKey(25)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Process all videos in a folder
def process_folder(folder_path):
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []

    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not video_files:
        print(f"No video files found in {folder_path}")
        return

    for video_file in video_files:
        print(f"Processing: {video_file}")
        process_video(video_file)

# Entry point
if __name__ == "__main__":
    folder_path = "DIP Project Videos"
    process_folder(folder_path)
