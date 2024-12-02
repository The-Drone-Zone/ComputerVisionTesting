import cv2
import numpy as np
import pandas as pd

class LidarAnalysis:
    def __init__(self):
        self.data = None

    def getScan(self, input):
        # Read the CSV file into a DataFrame
        self.data = pd.read_csv(input)
        return self.data

    def showImageScanPoints(self, img=[]):
        if len(img) == 0:
            # Create a blank image (black canvas)
            img_height, img_width = 500, 650
            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # Loop through each row
        for index, row in self.data.iterrows():
            img = cv2.circle(img, (int(row["x_position"]), int(row["y_position"])), 3, [255, 255, 0], -1)
            # Add depth to points
            if index % 13 == 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(round(row["distance"], 1)), (int(row["x_position"]), int(row["y_position"]) - 8), font, 0.4, (255, 255, 255), 2)

        cv2.imshow("Scan Points", img)

        cv2.waitKey(0)

    def showFrameScanPoints(self, img=[]):
        if len(img) == 0:
            # Create a blank image (black canvas)
            img_height, img_width = 500, 650
            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # Loop through each row
        for index, row in self.data.iterrows():
            img = cv2.circle(img, (int(row["x_position"]), int(row["y_position"])), 3, [255, 255, 0], -1)
            # Add depth to points
            # if index % 5 == 0:
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     cv2.putText(img, str(round(row["distance"], 2)), (int(row["x_position"]) + 5, int(row["y_position"]) + 5), font, 0.4, (255, 255, 255), 2)

        cv2.imshow("Scan Points", img)

if __name__ == '__main__':
    thing = LidarAnalysis()
    thing.getScan('lidar_reading_dataset.csv')
    thing.showImageScanPoints()