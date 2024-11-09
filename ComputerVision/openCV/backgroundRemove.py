import cv2
import numpy as np

cap = cv2.VideoCapture('ComputerVision/testVideos/video2.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # fgmask = fgbg.apply(frame)
    # cv2.imshow('Foreground Mask', fgmask)

    # # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # ret, threshold = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY)
    threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    # cv2.imshow('threshold', threshold)

    # Apply Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.dilate(threshold,kernel,iterations = 1)
    # morph = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    # Apply edge detection (Canny)
    edges = cv2.Canny(morph, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Draw the contours on the original image
    contoursImg = frame.copy()
    cv2.drawContours(contoursImg, contours, -1, (0, 255, 0), 2)
    contoursImg = cv2.cvtColor(contoursImg, cv2.COLOR_BGR2RGB)

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

    # # Display the image with contours
    cv2.imshow('Final', boxedImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
