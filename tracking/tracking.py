import datetime  # Library for handling date and time operations
from ultralytics import YOLO  # Library for loading and using the YOLO model
import cv2  # OpenCV library for image and video processing
from deep_sort_realtime.deepsort_tracker import DeepSort  # Library for the DeepSORT tracker


def create_video_writer(video_cap, output_filename):
    # Function to create a video writer object for saving the output video
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return writer


CONFIDENCE_THRESHOLD = 0.8  # Confidence threshold for detecting objects
GREEN = (0, 255, 0)  # Color for drawing bounding boxes
WHITE = (255, 255, 255)  # Color for drawing text

# Use webcam instead of a video file
video_cap = cv2.VideoCapture(0)  # Initialize video capture object to read from the webcam
# Optional: Initialize video writer if you want to save the output
writer = create_video_writer(video_cap, "output.mp4")

model = YOLO("yolov8n.pt")  # Load the pre-trained YOLOv8n model
tracker = DeepSort(max_age=50)  # Initialize the DeepSORT tracker

while True:
    start = datetime.datetime.now()  # Record the start time

    ret, frame = video_cap.read()  # Read a frame from the webcam
    if not ret:
        break  # Exit the loop if no frame is read

    detections = model(frame)[0]  # Run the YOLO model on the frame to detect objects
    results = []

    for data in detections.boxes.data.tolist():
        confidence = data[4]  # Extract the confidence level of the detection
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue  # Ignore detections with low confidence

        # Get the bounding box coordinates and class ID
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    tracks = tracker.update_tracks(results, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue  # Ignore unconfirmed tracks

        track_id = track.track_id  # Get the track ID
        ltrb = track.to_ltrb()  # Get the bounding box coordinates
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        # Draw the bounding box and the track ID on the frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    end = datetime.datetime.now()  # Record the end time
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"  # Calculate and display the FPS
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # Display the frame using OpenCV's imshow
    cv2.imshow("Frame", frame)
    writer.write(frame)  # Optional: write to video

    if cv2.waitKey(1) == ord("q"):
        break  # Exit the loop if 'q' key is pressed

video_cap.release()  # Release the video capture object
writer.release()  # Release the video writer object
cv2.destroyAllWindows()  # Close all OpenCV windows
