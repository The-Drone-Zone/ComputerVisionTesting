import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import components 
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python.components.containers import detections
from mediapipe.tasks.python.components.containers.detections import DetectionResult
import time

MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0) # red

class MobileNetV2:
    def __init__(self):
        # Models
        self.int8Model = './models/ssd_mobilenet_v2_int_8.tflite'
        self.float32Model = './models/ssd_mobilenet_v2_float_32.tflite'

        self.detector = None

        # Detection Results
        self.results = None

    def visualize(self, image) -> np.ndarray:
        if self.results != None:
            for detection in self.results.detections:
                # Draw bounding box
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

                # Draw Label abd score
                category = detection.categories[0]
                category_name = category.category_name
                probability = round(category.score, 2)
                result_text = category_name + ' (' + str(probability) + ')'
                text_location = (MARGIN + bbox.origin_x,
                                MARGIN + ROW_SIZE + bbox.origin_y)
                cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return image

    def print_result(self, result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
        print('Detection results {}\n{}'.format(result, timestamp_ms))
        self.results = result

    def runLiveDetection(self):
        # Create an Object Detector object
        base_options = python.BaseOptions(model_asset_path=self.float32Model)
        runningMode = vision.RunningMode.LIVE_STREAM
        options = vision.ObjectDetectorOptions(base_options=base_options, 
                                            score_threshold = 0.2, 
                                            running_mode=runningMode, 
                                            result_callback=self.print_result)
        self.detector = vision.ObjectDetector.create_from_options(options)

        # Load image
        cap = cv2.VideoCapture(0)

        frame_index = 0
        timeSum = 0

        # Loop through video frames
        while cap.isOpened():
            # Read a frame from the video
                success, frame = cap.read()
                
                if success:
                    # Convert opencv image frame to mediapipe format
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                    # calculate timestamp
                    frame_index += 1
                    frame_timestamp_ms = int(1000 * frame_index / cap.get(cv2.CAP_PROP_FPS))
                    timer = time.time()

                    # Run detector
                    self.detector.detect_async(mp_image, frame_timestamp_ms)
                    
                    # Print time for detection
                    print('Processing time: {} ms'.format(round((time.time() - timer) * 1000, 3)))
                    timeSum += round((time.time() - timer) * 1000, 3)

                    image_copy = np.copy(mp_image.numpy_view())
                    annotated_image = self.visualize(image_copy)
                    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Detection', rgb_annotated_image)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
        print('Average processing time: {} ms'.format(round(timeSum / frame_index, 2)))

    def runVideoDetection(self):
        # Create an Object Detector object
        base_options = python.BaseOptions(model_asset_path=self.float32Model)
        runningMode = vision.RunningMode.VIDEO
        options = vision.ObjectDetectorOptions(base_options=base_options, 
                                            score_threshold = 0.2,
                                            running_mode=runningMode)
        self.detector = vision.ObjectDetector.create_from_options(options)

        # Load image
        cap = cv2.VideoCapture(0)

        frame_index = 0
        timeSum = 0

        # Loop through video frames
        while cap.isOpened():
            # Read a frame from the video
                success, frame = cap.read()
                
                if success:
                    # Convert opencv image frame to mediapipe format
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                    # calculate timestamp
                    frame_index += 1
                    frame_timestamp_ms = int(1000 * frame_index / cap.get(cv2.CAP_PROP_FPS))
                    timer = time.time()

                    # Run detector
                    self.results = self.detector.detect_for_video(mp_image, frame_timestamp_ms)
                    
                    # Print time for detection
                    print('Processing time: {} ms'.format(round((time.time() - timer) * 1000, 3)))
                    timeSum += round((time.time() - timer) * 1000, 3)

                    image_copy = np.copy(mp_image.numpy_view())
                    annotated_image = self.visualize(image_copy)
                    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Detection', rgb_annotated_image)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
        print('Average processing time: {} ms'.format(round(timeSum / frame_index, 2)))

if __name__ == '__main__':
    thing = MobileNetV2()
    # thing.runLiveDetection()
    thing.runVideoDetection()