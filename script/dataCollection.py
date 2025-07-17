import cv2
import numpy as np
import time
import csv
import math
from PIL import ImageGrab
import tobii_research as tr

# ---------------------------
# Use AppKit's NSScreen to get monitor information (macOS)
# ---------------------------
try:
    from AppKit import NSScreen

    screens = NSScreen.screens()
    if len(screens) < 2:
        raise Exception("Less than two monitors detected; cannot record second screen.")

    # Get the second monitor (external screen)
    second_screen = screens[1]
    frame = second_screen.frame()  # NSRect: origin and size

    # Get the main screen's height (primary screen) to convert coordinates:
    main_screen = screens[0]
    main_frame = main_screen.frame()
    main_height = main_frame.size.height

    # Convert AppKit coordinates (origin at bottom-left) to top-left origin for ImageGrab
    left = int(frame.origin.x)
    top = int(main_height - frame.origin.y - frame.size.height)
    right = int(frame.origin.x + frame.size.width)
    bottom = int(main_height - frame.origin.y)
    bbox = (left, top, right, bottom)
    screen_width = right - left
    screen_height = bottom - top

    print("Recording external screen with bbox:", bbox)
except Exception as e:
    print("Error obtaining external monitor info with AppKit NSScreen:", e)
    from pyautogui import size
    screen_width, screen_height = size()
    bbox = (0, 0, screen_width, screen_height)
    print("Falling back to primary screen:", bbox)

# ---------------------------
# Initialize Tobii Eye Tracker
# ---------------------------
found_eyetrackers = tr.find_all_eyetrackers()
if not found_eyetrackers:
    raise Exception("No eyetrackers found.")
my_eyetracker = found_eyetrackers[0]
print("Address: " + my_eyetracker.address)
print("Model: " + my_eyetracker.model)
print("Name: " + my_eyetracker.device_name)
print("Serial number: " + my_eyetracker.serial_number)

# Global variable to store the most recent gaze sample
latest_gaze_data = None

def gaze_data_callback(gaze_data):
    global latest_gaze_data
    latest_gaze_data = gaze_data
    # Uncomment for debugging:
    # print("Left eye: {}  Right eye: {}".format(
    #     gaze_data['left_gaze_point_on_display_area'],
    #     gaze_data['right_gaze_point_on_display_area']))

my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

# ---------------------------
# Initialize Face Video Capture
# ---------------------------
face_cap = cv2.VideoCapture(0)  # Change index if needed
if not face_cap.isOpened():
    raise Exception("Cannot open webcam.")

face_width = int(face_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
face_height = int(face_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
face_fps_target = 30  # target loop rate

# Instead of writing frames directly, store them with their capture absolute timestamps
face_frames = []    # Each element is (frame, abs_timestamp)
screen_frames = []  # Each element is (frame, abs_timestamp)

# ---------------------------
# Open CSV for Gaze Data Logging with additional fields
# ---------------------------
csv_file = open('gaze_data_*.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'AbsoluteTimestamp', 'RelativeTime', 
    'Left Gaze X', 'Left Gaze Y', 
    'Right Gaze X', 'Right Gaze Y', 'Gaze HZ',
    'Left Gaze Origin', 'Left Gaze Origin Validity',
    'Right Gaze Origin', 'Right Gaze Origin Validity',
    'Left Gaze Point (User Coord)', 'Left Gaze Point Validity',
    'Right Gaze Point (User Coord)', 'Right Gaze Point Validity'
])

# ---------------------------
# Start Recording with a Common Reference Time
# ---------------------------
start_time = time.time()
record_duration = 300  # seconds; adjust as needed
frame_interval = 1.0 / face_fps_target  # target interval between iterations
next_frame_time = time.time()

print("Recording started... Press 'q' in the video window to stop early.")

while True:
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > record_duration:
        break

    # Enforce 30 Hz loop rate
    if current_time < next_frame_time:
        time.sleep(next_frame_time - current_time)
    next_frame_time += frame_interval

    abs_timestamp = time.time()
    rel_time = abs_timestamp - start_time

    # ---------------------------
    # Retrieve and Validate Gaze Data
    # ---------------------------
    if latest_gaze_data is not None:
        left_gaze = latest_gaze_data.get('left_gaze_point_on_display_area')
        right_gaze = latest_gaze_data.get('right_gaze_point_on_display_area')
        left_origin = latest_gaze_data.get('left_gaze_origin_in_user_coordinate_system')
        left_origin_validity = latest_gaze_data.get('left_gaze_origin_validity')
        right_origin = latest_gaze_data.get('right_gaze_origin_in_user_coordinate_system')
        right_origin_validity = latest_gaze_data.get('right_gaze_origin_validity')
        left_point_user = latest_gaze_data.get('left_gaze_point_in_user_coordinate_system')
        left_point_validity = latest_gaze_data.get('left_gaze_point_validity')
        right_point_user = latest_gaze_data.get('right_gaze_point_in_user_coordinate_system')
        right_point_validity = latest_gaze_data.get('right_gaze_point_validity')
        
        # Validate gaze point on display data
        if (left_gaze is None or right_gaze is None or
            math.isnan(left_gaze[0]) or math.isnan(left_gaze[1]) or
            math.isnan(right_gaze[0]) or math.isnan(right_gaze[1])):
            left_gaze = (0.5, 0.5)
            right_gaze = (0.5, 0.5)
        # Validate origin data; default to center if missing
        if left_origin is None:
            left_origin = (0.5, 0.5)
        if right_origin is None:
            right_origin = (0.5, 0.5)
        # Validate gaze point in user coordinate system data; default to center if missing
        if left_point_user is None:
            left_point_user = (0.5, 0.5)
        if right_point_user is None:
            right_point_user = (0.5, 0.5)
        # Validate validity fields; default to 0 if missing
        if left_origin_validity is None:
            left_origin_validity = 0
        if right_origin_validity is None:
            right_origin_validity = 0
        if left_point_validity is None:
            left_point_validity = 0
        if right_point_validity is None:
            right_point_validity = 0
    else:
        left_gaze = (0.5, 0.5)
        right_gaze = (0.5, 0.5)
        left_origin = (0.5, 0.5)
        right_origin = (0.5, 0.5)
        left_origin_validity = 0
        right_origin_validity = 0
        left_point_user = (0.5, 0.5)
        right_point_user = (0.5, 0.5)
        left_point_validity = 0
        right_point_validity = 0

    csv_writer.writerow([
        abs_timestamp, rel_time,
        left_gaze[0], left_gaze[1],
        right_gaze[0], right_gaze[1],
        face_fps_target,
        left_origin, left_origin_validity,
        right_origin, right_origin_validity,
        left_point_user, left_point_validity,
        right_point_user, right_point_validity
    ])

    # ---------------------------
    # Capture Face Frame and store with absolute timestamp
    # ---------------------------
    ret, face_frame = face_cap.read()
    if not ret:
        print("Failed to capture face frame.")
        break
    face_frames.append((face_frame, abs_timestamp))

    # ---------------------------
    # Capture and Annotate Screen Frame (from external monitor) and store with absolute timestamp
    # ---------------------------
    screen = ImageGrab.grab(bbox=bbox)
    screen_frame = np.array(screen)
    screen_frame = cv2.cvtColor(screen_frame, cv2.COLOR_RGB2BGR)

    avg_gaze_x = (left_gaze[0] + right_gaze[0]) / 2.0
    avg_gaze_y = (left_gaze[1] + right_gaze[1]) / 2.0

    try:
        gaze_px = int(avg_gaze_x * screen_width)
        gaze_py = int(avg_gaze_y * screen_height)
    except Exception as e:
        print("Error converting gaze coordinates:", e)
        gaze_px = int(0.5 * screen_width)
        gaze_py = int(0.5 * screen_height)

    cv2.circle(screen_frame, (gaze_px, gaze_py), 15, (0, 0, 255), -1)
    screen_frames.append((screen_frame, abs_timestamp))

    cv2.imshow('External Screen with Gaze Overlay', screen_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------
# Cleanup live capture resources
# ---------------------------
my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
face_cap.release()
csv_file.close()
cv2.destroyAllWindows()

# ---------------------------
# Calculate Output Frame Rates (normal speed):
# Final video duration will be record_duration (e.g., 300 seconds for 300s recording)
num_face_frames = len(face_frames)
num_screen_frames = len(screen_frames)
output_face_fps = num_face_frames / record_duration
output_screen_fps = num_screen_frames / record_duration

print("Captured {} face frames. Output face fps: {:.2f}".format(num_face_frames, output_face_fps))
print("Captured {} screen frames. Output screen fps: {:.2f}".format(num_screen_frames, output_screen_fps))

# ---------------------------
# Write Face Video with Absolute Timestamp Overlay
# ---------------------------
face_video_writer = cv2.VideoWriter('face_video_*.mp4',
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    output_face_fps, (face_width, face_height))
for (frame, timestamp) in face_frames:
    # Format absolute system time as human-readable
    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    text = "Time: " + timestr
    cv2.putText(frame, text, (10, face_height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    face_video_writer.write(frame)
face_video_writer.release()

# ---------------------------
# Write Screen Video with Absolute Timestamp Overlay
# ---------------------------
screen_video_writer = cv2.VideoWriter('screen_video_*.mp4',
                                      cv2.VideoWriter_fourcc(*'mp4v'),
                                      output_screen_fps, (screen_width, screen_height))
for (frame, timestamp) in screen_frames:
    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    text = "Time: " + timestr
    cv2.putText(frame, text, (10, screen_height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    screen_video_writer.write(frame)
screen_video_writer.release()

print("Recording finished.")
