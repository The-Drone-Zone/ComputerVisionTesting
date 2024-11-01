import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time

def tracking():
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Average time
    timeSum = 0
    frameCounter = 0

    # Loop through the video frames
    while cap.isOpened():
        # start timer
        timer = time.time()

        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True) # Track (better for videos?)
            # results = model(frame) # Inference (For images)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Time per frame
            currTime = round((time.time() - timer) * 1000, 2)
            print('Time for frame: {}'.format(currTime))
            timeSum += currTime
            frameCounter += 1

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    print('Average time: {}'.format(timeSum / frameCounter))

def plotOverTime():
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            if results[0].boxes.id != None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y), float(w), float(h)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                # points = np.hstack(track[:2]).astype(np.int32).reshape((-1, 1, 2))
                # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                if track[0][2] > w and track[0][3] > h:
                    print(f'object: {track_id} is getting FURTHER')
                elif track[0][2] < w and track[0][3] < h:
                    print(f'object: {track_id} is getting CLOSER')

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def imageDetection():
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Open the camera
    image = cv2.imread('ComputerVision/testImages/img3.jpg')

    # Run YOLOv8 model detection
    results = model(image) # Inference (For images)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Break the loop
    cv2.waitKey(0)

    # close the display window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # plotOverTime()
    # tracking()
    imageDetection()