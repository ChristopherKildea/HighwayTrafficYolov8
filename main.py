import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

model = YOLO('yolov8n.pt')
video_path = "./test.mp4"
cap = cv2.VideoCapture(video_path)

# Specifies at what time intervals (in seconds) traffic data is collected over
DATA_COLLECTION_INTERVAL = 8

# The road's speed limit and the distance between the two marked lines
SPEED_LIMIT = 66  # 45 mph = 66 ft/sec
DIST = 18.4  # feet

# Where vehicles should begin to be tracked
TRACKING_BEGIN = 700
# Where vehicles should cease to be tracked
TRACKING_END = 900

# Store the number of frames each vehicle was
# between the lines
vehicle_frames = {}

# Store the time each vehicle was between the lines
vehicle_times = []

# Get the fps of the video for time calculation
FPS = cap.get(cv2.CAP_PROP_FPS)

percent_diff_list = []


def graph_data(graph_list):
    """
    Plots and graphs the given data

    :param graph_list: A list of data points to plot
    """

    time = [item[0] for item in graph_list]
    percent_difference = [item[1] for item in graph_list]

    if len(time) != 0:

        # Create/plot points
        plt.figure(figsize=(16, 6))
        ax = plt.gca()
        ax.plot(time, percent_difference, marker='o')

        # Add labels and title
        ax.set_xlabel('Time (Sec)')
        ax.set_ylabel('Traffic Percentage')
        ax.set_title('Time vs Traffic Percentage')

        # Create table data/table
        table_data = np.column_stack([time, np.round(percent_difference, 1)])
        table = ax.table(cellText=table_data, colLabels=['Time', 'Percent Difference'], loc='center right',
                         bbox=[1.1, 0.1, 0.2, 0.8])

        # Adjust properties of tables for better layout
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        plt.tight_layout()

        plt.show()
    else:
        # When an empty list has been passed in and no data is available
        # to be graphed

        # Display a message stating that not enough time was given for data
        # collection
        text = "Video has not reached minimum time interval for data collection"
        print(text)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.text(0.5, 0.5, text, fontsize=20, ha='center', color='red')

        plt.show()


def record_traffic_percentage(times_list, time):
    """
    Average a list of times and calculate the percentage of traffic present

    :param times_list: A list of times the vehicles were between the lines
    :param time: The time at which this data corresponds to
    """

    # The average of all the times
    avg_time = sum(times_list) / len(times_list)

    # The time it would take to drive between
    # the two lines if going the speed limit
    expected_time = DIST / SPEED_LIMIT
    # The increased time taken to cross between the lines due to traffic
    percent_diff = ((avg_time - expected_time) / expected_time) * 100

    # If cars are going above the speed limit, there is no traffic
    # Make the traffic percentage 0
    if percent_diff < 0:
        percent_diff = 0

    percent_diff_list.append([time, percent_diff])  # 1 will be replaced whatever the correct time is


def analyze_video(by_frame):
    """
    Goes through a video and tracks the speed of cars/trucks between
    the two lines

    :param by_frame: "True" to manually iterate the video frame-by-frame, "False" to
    allow the video to play out
    :return:
    """

    total_time_requirement = DATA_COLLECTION_INTERVAL

    # The number of frames into the video at a given time
    frame_count = 0

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:

            # Manually iterate through the video frame-by-frame if by_frame is True
            if by_frame:
                if cv2.waitKey(0) & 0xFF == ord('n'):
                    continue

            # Draw lines displaying starting and stopping points for checking data
            cv2.line(frame, (0, TRACKING_BEGIN), (2000, TRACKING_BEGIN), (0, 255, 0), 3)
            cv2.line(frame, (0, TRACKING_END), (2000, TRACKING_END), (0, 0, 255), 3)

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=0.1, classes=[2,7])

            if results[0] is not None and results[0].boxes.id is not None:

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
                vid_time = frame_count / FPS

                print(f"Time: {vid_time}")
                print(f"Vehicle frames dict: {vehicle_frames}")

                # Plot the tracks
                for box, vehicle_id in zip(boxes, vehicle_ids):

                    # Dimensions of the bounding box
                    x, y, w, h = box

                    # Add the given vehicle to the list
                    if y <= TRACKING_BEGIN and vehicle_frames.get(vehicle_id) is None:

                        vehicle_frames[vehicle_id] = 0

                    # Check if the box has passed the first line
                    elif y > TRACKING_BEGIN and vehicle_frames.get(vehicle_id) is not None:

                        if y < TRACKING_END:  # The box is in between the first and seconds lines; add to its frame history

                            vehicle_frames[vehicle_id] += 1
                        else:

                            # Record the time that the vehicle was between the lines and delete its index in the dictionary
                            print(f"send data for {vehicle_id}")
                            vehicle_times.append(vehicle_frames[vehicle_id] / FPS)
                            del vehicle_frames[vehicle_id]

                # If we have surpassed our time interval, process
                # the data for the given time interval
                if vid_time > total_time_requirement:

                    # If there is data to collect
                    if len(vehicle_times) > 0:
                        print("Sending vehicle_times data")
                        print(vehicle_times)

                        # Process the data
                        record_traffic_percentage(vehicle_times, total_time_requirement)
                        vehicle_times.clear()
                    else:
                        print("No data to record in this time interval")

                    # Calculate the next time to collect data
                    total_time_requirement += DATA_COLLECTION_INTERVAL

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

    return percent_diff_list


def main():
    traffic_data = analyze_video(False)
    graph_data(traffic_data)


if __name__ == "__main__":
    main()
