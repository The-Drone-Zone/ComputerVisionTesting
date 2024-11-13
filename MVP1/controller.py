from imageAnalysis import ImageAnalysis
from dataDisplay import DataDisplay
import cv2
import time

def runImage(imgPath):
    imageAnalysis = ImageAnalysis()
    obstacles = imageAnalysis.processImage(imgPath)
    display = DataDisplay()
    display.plotImage(obstacles)

def runVideo():
    imageAnalysis = ImageAnalysis()
    imageAnalysis.initCamera()

    display = DataDisplay()

    # Average time
    timeSum = 0
    frameCounter = 0

    while True:
        # start timer
        timer = time.time()
        obstacles = imageAnalysis.processVideoFrame()
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
    runVideo()