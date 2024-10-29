import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time

def tracking():
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Open the camera
    cap = Picamera2()

    # Average time
    timeSum = 0
    frameCounter = 0

    # Loop through the video frames
    while True:
        # start timer
        timer = time.time()

        # Read a frame from the video
        frame = cap.capture_array()

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
    # cap.release()
    cv2.destroyAllWindows()
    print('Average time: {}'.format(timeSum / frameCounter))

if __name__ == '__main__':
    # plotOverTime()
    tracking()