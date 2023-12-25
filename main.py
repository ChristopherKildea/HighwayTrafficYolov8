from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "./test.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: 0)
# List of times a given object was in the video
in_vid_list = []

fps = cap.get(cv2.CAP_PROP_FPS)

# Filter results so they only include those within the specified range
"""
def filter_results(results):

    to_display = []
    # x, y, w, h = box

    for box in results.boxes:
        print("----------------")
        print(box)
        print("----------------")

        y_center = box.xywh[0][1]

        if tracking_begin < y_center < tracking_end:
            to_display.append(box)
    return to_display
"""

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        # must press a to advance ot next frame
        if cv2.waitKey(0) & 0xFF == ord('n'):
            continue


        # adding this comment due to github

        tracking_begin = 300
        tracking_end = 500
        cv2.line(frame, (0, tracking_begin), (2000, tracking_begin), (0, 255, 0), 3)
        cv2.line(frame, (0, tracking_end), (2000, tracking_end), (0, 0, 255), 3)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.45)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        #annotated_frame = filter_results(results[0])
        # Display the annotated frame
        annotated_frame = cv2.resize(annotated_frame, (960,540))
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):

            x, y, w, h = box



            # Check if the box is in range
            if y > tracking_begin:  # in between the lines

                print(f"Passed tracking_begin: {track_id}")

                if y < tracking_end:  # in between lines, record
                    print(f"Between lines: {track_id}")
                    track_history[track_id] += 1
                elif y > tracking_end and track_history[track_id] != 0:  # Below ending line and hasn't been recorded
                    # calculate/record time
                    print(f"Passed tracking_end: {track_id}")
                    in_vid_length = track_history[track_id] / fps
                    print(f"The in_vid_length of {track_id} is {in_vid_length}")
                    in_vid_list.append(in_vid_length)

                    # set its value in the list to 0
                    track_history[track_id] = 0















        # we can loop through the object's bounding boxes and see if they are below a certain point


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Q has been pressed")
            break
    else:
        # Break the loop if the end of the video is reached
        print("Video end reached")
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()