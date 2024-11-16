from imageAnalysis import ImageAnalysis
from dataDisplay import DataDisplay
import cv2
import time

def runImage(imgPath):
    imageAnalysis = ImageAnalysis()
    obstacles = imageAnalysis.processImage(imgPath)
    display = DataDisplay()
    display.plotImage(obstacles)

def runVideo(input):
    imageAnalysis = ImageAnalysis()
    imageAnalysis.initCamera(input)

    display = DataDisplay()

    imageTracker = []

    # Average time
    timeSum = 0
    frameCounter = 0

    while True:
        # start timer
        timer = time.time()
        
        obstacles = imageAnalysis.processVideoFrame()

        # WIP for tracking (camera)
        # for obstacle in imageTracker list
            # optical flow (for single obstacle)
            # if points matched > zero and still within bounds (10-20m box)
                # increase frames detected count += 1
                # if frames detected count >= 5 
                    # STOP DRONE
                    # remove from imageTracker list
            # else
                # remove from imageTracker list
        # for obstacle in obstacles
            # if within certain bounds of camera (10-20m box)
                # featureDetection (pass cropped image OR crop in function)
                # set frames detected count = 1
                # append to imageTracker list

        display.plotVideoFrame(obstacles)

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
    # runImage('ComputerVision/testImages/img2.jpg')
    runVideo('ComputerVision/testVideos/video3.mp4')
    # runVideo(0)