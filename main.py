from collections import defaultdict
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np


# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "./test.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
# i.e. the number of frames the given vehicle was
# between the beginning and end lines
vehicle_frames = defaultdict(lambda: 0)
# List that gives the amount of times to go from the beginning ot the end liens
# for each vehicle
vehicle_times = []
# Get the fps of the video for time calculation
fps = cap.get(cv2.CAP_PROP_FPS)

# The current number of frames gone over
frame_count = 0

# Specifies at what intervals traffic data is collected over
data_collection_interval = 1
total_time_requirement = data_collection_interval

percent_diff_list = []


def graph_data(graph_list):
    time = [item[0] for item in graph_list]
    percent_difference = [item[1] for item in graph_list]

    plt.figure(figsize=(16, 6))
    ax = plt.gca()  # Getting the current axes

    ax.plot(time, percent_difference, marker='o')  # Plotting the data points

    # Adding labels and title
    ax.set_xlabel('Time (Sec)')
    ax.set_ylabel('Percent Difference')
    ax.set_title('Time vs Percent Difference')

    # Creating the table data
    table_data = np.column_stack([time, np.round(percent_difference, 1)])

    # Manually setting the position of the table relative to the plot's axes
    # The bbox argument defines the position: [left, bottom, width, height]
    table = ax.table(cellText=table_data, colLabels=['Time', 'Percent Difference'], loc='center right',
                     bbox=[1.1, 0.1, 0.2, 0.8])

    # Adjusting table properties for better layout
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Adjusting plot layout to ensure proper display
    plt.tight_layout()

    # Display the plot
    plt.show()

def record_traffic_percentage(times_list, time):

    avg_time = sum(times_list) / len(times_list)

    speed_limit = 66  # 45 mph = 66 ft/sec
    dist = 35  # feet

    # The time to go the distance if going the speed limit
    expected_time = dist / speed_limit

    percent_diff = ((avg_time - expected_time) / expected_time) * 100

    # Make traffic 0% if cars are going above the speed limit
    if percent_diff < 0:
        percent_diff = 0

    percent_diff_list.append([time, percent_diff])  # 1 will be replaced whatever the correct time is







while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        # must press a to advance ot next frame
        """
        if cv2.waitKey(0) & 0xFF == ord('n'):
            continue
        """

        tracking_begin = 600
        tracking_end = 900

        # Draw lines displaying starting and stopping points for checking data
        cv2.line(frame, (0, tracking_begin), (2000, tracking_begin), (0, 255, 0), 3)
        cv2.line(frame, (0, tracking_end), (2000, tracking_end), (0, 0, 255), 3)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.1, classes=[2,7])


        if results[0] != None and results[0].boxes.id != None:


            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            vehicle_ids = results[0].boxes.id.int().cpu().tolist()



            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            annotated_frame = cv2.resize(annotated_frame, (960,540))
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Find the current time on the video
            frame_count += 1
            vid_time = frame_count / fps  # works

            # Plot the tracks
            for box, vehicle_id in zip(boxes, vehicle_ids):

                # Dimensions of the bounding box
                x, y, w, h = box

                # Check if the box has passed the first line
                if y > tracking_begin:

                    # print(f"Passed tracking_begin: {track_id}")

                    if y < tracking_end:  # The box is in between the first and seconds lines; add to its frame history
                        # print(f"Between lines: {track_id}")
                        vehicle_frames[vehicle_id] += 1
                    elif y > tracking_end and vehicle_frames[vehicle_id] != 0:  # Below ending line and hasn't been recorded
                        # calculate/record time
                        # print(f"Passed tracking_end: {track_id}")
                        # Calculate the time the vehicle took to cross from the beginning to the end line
                        # based on the number of frames it was between the lines and the fps

                        vehicle_times.append(vehicle_frames[vehicle_id] / fps)

                        # set its value in the list to 0
                        vehicle_frames[vehicle_id] = 0

            # If we have surpassed our time interval, process
            # the data for the given time interval
            if vid_time > total_time_requirement:

                # If there is data to collect
                if len(vehicle_times) > 0:

                    # This works

                    # print(f"total_time_requirement: {total_time_requirement}")


                    # Find average

                    # Process the data
                    record_traffic_percentage(vehicle_times, total_time_requirement)
                    vehicle_times.clear()


                # Calculate the next time to collect data
                total_time_requirement += data_collection_interval

        # we can loop through the object's bounding boxes and see if they are below a certain point
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Q has been pressed")
            break
    else:
        # Break the loop if the end of the video is reached
        print("Video end reached")
        break

# graph and display data
graph_data(percent_diff_list)



# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()