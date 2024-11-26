import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

def imageMorphology():
    # Display the images with matplotlib
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("Image Processing Steps", fontsize=16)

    frame = cv2.imread('ComputerVision/testImages/img5.jpg')

    orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(orig)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # axes[0, 0].imshow(gray, cmap="gray")
    # axes[0, 0].set_title("Grayscale")
    # axes[0, 0].axis("off")

    # # ret, threshold = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY)
    threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 75, 15)
    # cv2.imshow('threshold', threshold)
    axes[0, 1].imshow(threshold, cmap="gray")
    axes[0, 1].set_title("Threshold")
    axes[0, 1].axis("off")

    # Apply Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.dilate(threshold,kernel,iterations = 1)
    # morph = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    axes[0, 2].imshow(morph, cmap="gray")
    axes[0, 2].set_title("Dilation")
    axes[0, 2].axis("off")

    # Apply edge detection (Canny)
    edges = cv2.Canny(morph, 100, 200)
    axes[1, 0].imshow(edges, cmap="gray")
    axes[1, 0].set_title("Edges")
    axes[1, 0].axis("off")

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Draw the contours on the original image
    contoursImg = frame.copy()
    cv2.drawContours(contoursImg, contours, -1, (0, 255, 0), 2)
    contoursImg = cv2.cvtColor(contoursImg, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(contoursImg)
    axes[1, 1].set_title("Contours")
    axes[1, 1].axis("off")

    # Draw bounding boxes
    boxedImage = frame.copy()
    for cnt in contours:
        # poly = cv2.approxPolyDP(cnt, 3, True)
        # cnt = cv2.convexHull(cnt)
        # x,y,w,h = cv2.boundingRect(cnt)
        # if (w * h > 0):
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)),2)
            # cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        rect = cv2.minAreaRect(cnt)
        # Check if width * height > minArea
        if rect[1][0] * rect[1][1] > 250:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(boxedImage,[box],0,(0,0,255),2)
    
    cv2.imshow('Final', boxedImage)
    final = cv2.cvtColor(boxedImage, cv2.COLOR_BGR2RGB)
    axes[1, 2].imshow(final)
    axes[1, 2].set_title("Final Boxes")
    axes[1, 2].axis("off")

    # # Display the image with contours
    # cv2.imshow('Contours', frame)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    # cv2.waitKey(0)
    
    # cv2.destroyAllWindows()

def videoMorphology():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # ret, threshold = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY)
        threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
        # cv2.imshow('threshold', threshold)

        # Apply Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.dilate(threshold,kernel,iterations = 1)
        # morph = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        # morph = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('morphology', morph)

        # Apply edge detection (Canny)
        edges = cv2.Canny(morph, 100, 200)
        # cv2.imshow('edges', edges)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Draw the contours on the original image
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Draw bounding boxes
        for cnt in contours:
            # poly = cv2.approxPolyDP(cnt, 3, True)
            # cnt = cv2.convexHull(cnt)
            # x,y,w,h = cv2.boundingRect(cnt)
            # if (w * h > 0):
                # cv2.rectangle(frame,(x,y),(x+w,y+h),(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)),2)
                # cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
            rect = cv2.minAreaRect(cnt)
            # Check if width * height > minArea
            if rect[1][0] * rect[1][1] > 250:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame,[box],0,(0,0,255),2)

        # # Display the image with contours
        # cv2.imshow('Final', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    imageMorphology()
    # videoMorphology()