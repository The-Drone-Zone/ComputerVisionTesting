from imageAnalysis import ImageAnalysis
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random
import numpy as np
import cv2

class DataDisplay:

    def plotVideoFrame(self, obstacles):
        # Create a blank image (black canvas)
        img_height, img_width = 500, 650
        blank_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        for obstacle in obstacles:
            # Create bounding box
            cv2.drawContours(blank_img,[obstacle.corners],0,(0,0,255),2)
            # Add depth annotation at the center (CHANGE TO HAVE LIDAR DEPTH)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(blank_img, str(random.randint(0, 40)), (obstacle.x, obstacle.y), font, 1, (255, 255, 255), 2)

        cv2.imshow('Analysis Results', blank_img)

    def plotImage(self, obstacles):
        # Create a figure and axis
        fig, ax = plt.subplots()

        for obstacle in obstacles:
            # Create bounding box as polygon
            rect = Polygon(obstacle.corners, closed=True, edgecolor='red', facecolor='none', linewidth=2)
            # Add polygons to the plot
            ax.add_patch(rect)
            # Add depth annotation at the center (CHANGE TO HAVE LIDAR DEPTH)
            ax.text(obstacle.x, obstacle.y, f"{random.randint(0, 40)}", color='blue', fontsize=12, ha='center', va='center')

        # Set axis limits
        ax.autoscale()

        # Invert the y-axis to match OpenCV's coordinate system
        ax.invert_yaxis()

        # Add grid and show plot
        ax.grid(True)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Image Analysis Results")
        plt.show()

    def run(self):
        imageAnalysis = ImageAnalysis()
        obstacles = imageAnalysis.processImage('ComputerVision/testImages/img1.jpg')
        self.plotImage(obstacles)

if __name__ == '__main__':
    display = DataDisplay()
    display.run()
