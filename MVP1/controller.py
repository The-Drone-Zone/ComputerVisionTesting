from imageAnalysis import ImageAnalysis
from lidarAnalysis import LidarAnalysis
from dataDisplay import DataDisplay
import cv2
import time
import pandas as pd

def mapImageLidar(imageObstacles, lidarPoints):
    for obstacle in imageObstacles:
        # Compute the bounding rectangle once for this obstacle
        x, y, w, h = cv2.boundingRect(obstacle.corners)

        # Use pandas filtering to find points within the bounding box
        filtered_points = lidarPoints[
            (lidarPoints["x_position"] >= x) &
            (lidarPoints["x_position"] <= x + w) &
            (lidarPoints["y_position"] >= y) &
            (lidarPoints["y_position"] <= y + h)
        ]

        # Find the minimum distance from the filtered points
        if not filtered_points.empty:
            min_distance = filtered_points["distance"].min()
            if obstacle.distance == -1 or min_distance < obstacle.distance:
                obstacle.distance = min_distance


def runImage(imgPath, scanPath):
    # Initialization
    imageAnalysis = ImageAnalysis()
    display = DataDisplay()
    lidarAnalysis = LidarAnalysis()

    # Get and Process Data
    imageObstacles = imageAnalysis.processImage(imgPath)
    lidarPoints = lidarAnalysis.getScan(scanPath)

    # Map image and lidar together
    mapImageLidar(imageObstacles, lidarPoints)

    lidarAnalysis.showImageScanPoints(imageAnalysis.original)
    display.plotImage(imageObstacles)

def runVideo(videoIn, lidarIn):
    imageAnalysis = ImageAnalysis()
    display = DataDisplay()
    lidarAnalysis = LidarAnalysis()

    imageAnalysis.initCamera(videoIn)

    # For tracking image features across frames
    imageTracker = []
    screen_bounds = (200, 100, 450, 200)
    x_min, y_min, x_max, y_max = screen_bounds

    # Average time
    timeSum = 0
    frameCounter = 0

    while True:
        # start timer
        timer = time.time()
        
        imageObstacles = imageAnalysis.processVideoFrame()
        lidarPoints = lidarAnalysis.getScan(lidarIn)
        # Map image and lidar together
        mapImageLidar(imageObstacles, lidarPoints)

        # WIP for tracking (camera)
        for obstacle in imageTracker:
            # optical flow (for single obstacle)
            imageAnalysis.opticalFlow(obstacle)
            # if points matched > zero and center still within bounds (10-20m box)
            if len(obstacle.points) > 0:
                rect_center, _, _ = cv2.minAreaRect(obstacle.points) # get center
                if x_min <= rect_center[0] <= x_max and y_min <= rect_center[1] <= y_max:
                    obstacle.trackCount += 1
                    if obstacle.trackCount >= 5:
                        print("STOP DRONE")
                        imageTracker.remove(obstacle)
                        continue
                else:
                    imageTracker.remove(obstacle)
                    print('remove out of bounds object')
            else:
                imageTracker.remove(obstacle)
                print('remove un-found object')
        for obstacle in imageObstacles:
            # if within certain bounds of camera (10-20m box)
            if (x_min <= obstacle.x <= x_max and y_min <= obstacle.y <= y_max):
                # does feature detection for one obstacle, returns new TrackedObject, appends to list
                trackedObject = imageAnalysis.featureDetection(obstacle)
                if len(trackedObject.points) > 0:
                    imageTracker.append(trackedObject)
                    # set frames detected count = 1 (DONE ON TrackedObject initialization)

        lidarAnalysis.showFrameScanPoints(imageAnalysis.original)
        # imageAnalysis.displayFrame('rgb')
        display.plotVideoFrame(imageObstacles)
        imageAnalysis.old_frame = imageAnalysis.frame.copy()

        # Time per frame
        currTime = round((time.time() - timer) * 1000, 2)
        print('Time for frame (ms): {}'.format(currTime))
        timeSum += currTime
        frameCounter += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print('Average time (ms): {}'.format(timeSum / frameCounter))

if __name__ == '__main__':
    imagePath = 'ComputerVision/testImages/img2.jpg'
    videoPath = 'ComputerVision/testVideos/video3.mp4'
    lidarDataPath = 'poly_lidar_reading_dataset.csv'

    # runImage(imagePath, lidarDataPath)
    runVideo(videoPath, lidarDataPath)
    # runVideo(0, lidarDataPath)