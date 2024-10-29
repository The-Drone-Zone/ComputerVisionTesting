import cv2

class Contours:
    def detectContours(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():

            # Load the frame
            success, frame = cap.read()

            if success:
                # Convert the image to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply edge detection (Canny)
                edges = cv2.Canny(gray, 100, 200)

                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw the contours on the original image
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

                # Display the image with contours
                cv2.imshow('Contours', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def boxedContours(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():

            # Load the frame
            success, frame = cap.read()

            if success:
                # Convert the image to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply edge detection (Canny)
                edges = cv2.Canny(gray, 100, 200)

                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    # poly = cv2.approxPolyDP(cnt, 3, True)
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                # Draw the contours on the original image
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

                # Display the image with contours
                cv2.imshow('Contours', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def mergedBoxedContours(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            # Load the frame
            success, frame = cap.read()

            if success:
                # Convert the image to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply edge detection (Canny)
                edges = cv2.Canny(gray, 100, 200)

                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # List to store all bounding boxes
                bounding_boxes = []

                # Get bounding boxes for each contour
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    bounding_boxes.append([x, y, w, h])

                # Function to check if two boxes overlap
                def overlap(box1, box2):
                    x1, y1, w1, h1 = box1
                    x2, y2, w2, h2 = box2

                    return not (x1 > x2 + w2 or x1 + w1 < x2 or y1 > y2 + h2 or y1 + h1 < y2)

                # Merge overlapping boxes
                merged_boxes = []
                while bounding_boxes:
                    box = bounding_boxes.pop(0)
                    for other_box in bounding_boxes[:]:
                        if overlap(box, other_box):
                            # Merge the boxes
                            x1, y1, w1, h1 = box
                            x2, y2, w2, h2 = other_box
                            new_x = min(x1, x2)
                            new_y = min(y1, y2)
                            new_w = max(x1 + w1, x2 + w2) - new_x
                            new_h = max(y1 + h1, y2 + h2) - new_y
                            box = [new_x, new_y, new_w, new_h]
                            bounding_boxes.remove(other_box)  # Remove merged box
                    merged_boxes.append(box)

                # Draw the merged bounding boxes
                for (x, y, w, h) in merged_boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Display the image with contours
                cv2.imshow('Contours', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def thresholdContours(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():

            # Load the frame
            success, frame = cap.read()

            if success:
                # Convert the image to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                ret, threshold = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY)
                # cv2.imshow('threshold', threshold)

                # Apply edge detection (Canny)
                edges = cv2.Canny(threshold, 100, 200)
                # cv2.imshow('Canny', edges)

                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    poly = cv2.approxPolyDP(cnt, 3, True)
                    x,y,w,h = cv2.boundingRect(poly)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                # Draw the contours on the original image
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

                # Display the image with contours
                cv2.imshow('Contours', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def adaptThresholdContours(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():

            # Load the frame
            success, frame = cap.read()

            if success:
                # Convert the image to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)
                # cv2.imshow('threshold', threshold)

                # Apply edge detection (Canny)
                edges = cv2.Canny(threshold, 100, 200)
                # cv2.imshow('Canny', edges)

                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    poly = cv2.approxPolyDP(cnt, 3, True)
                    x,y,w,h = cv2.boundingRect(poly)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                # Draw the contours on the original image
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

                # Display the image with contours
                cv2.imshow('Contours', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    thing = Contours()
    # thing.detectContours()
    # thing.boxedContours()
    thing.mergedBoxedContours()
    # thing.thresholdContours()
    # thing.adaptThresholdContours()