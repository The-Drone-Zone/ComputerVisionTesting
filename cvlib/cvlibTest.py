import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    bbox, label, conf = cv.detect_common_objects(frame, confidence=0.1, model='yolov3-tiny')
    output_image = draw_bbox(frame, bbox, label, conf)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break