import cv2
import numpy as np
from matplotlib import pyplot as plt
from obstacle import BoundedObstacle, TrackedObject

class ImageAnalysis:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.old_frame = None
        self.original = None

        # Initialize FAST feature detection object
        self.fast = cv2.FastFeatureDetector_create()

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def initCamera(self, input):
        self.cap = cv2.VideoCapture(input)

    def releaseCamera(self):
        self.cap.release()

    def loadFrame(self):
        success, self.frame = self.cap.read()
        self.original = self.frame.copy()

        if not success:
            print('Failed to read frame')
            exit()
    
    def loadImage(self, path):
        self.frame = cv2.imread(path)
        self.original = self.frame.copy()

    def grayscale(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def blur(self):
        # Use bilateral filter to reduce noise while keeping edges sharp
        # d = number of neighboring pixels to include
        # sigmacolor is the Standard deviation in the color space.
        #sigmaSpace is the Standard deviation in the coordinate space (in pixel terms)
        self.frame = cv2.bilateralFilter(self.frame, d=11, sigmaColor=12, sigmaSpace=50)

        # Use GaussianBlur to reduce noise
        # The size of the kernel to be used (the neighbors to be considered). w and h have to be odd and positive numbers
        # The standard deviation in x. Writing 0 implies that sigma is calculated using kernel size.
        # The standard deviation in y. Writing 0 implies that sigma is calculated using kernel size.
        # self.frame = cv2.GaussianBlur(self.frame, (5, 5), sigmaX=0)

        
    def threshold(self):
        #Non-zero value assigned to the pixels for which the condition is satisfied
        #Adaptive thresholding algorithm to use, see AdaptiveThresholdTypes
        #Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV
        self.frame = cv2.adaptiveThreshold(self.frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 75, 15)

        #_, self.frame = cv2.threshold(self.frame, 150, 255, cv2.THRESH_BINARY_INV) #has trouble with sky

    def morphology(self):
        # get kernel of 1s or size specified (2x2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        self.frame = cv2.dilate(self.frame, kernel, iterations = 1)

        # erosion then dilation
        # morph = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        # dilation then erosion
        # morph = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    def edgeDetection(self):
        #threshold1 below this not an edge
        #threshold2 above this defintly an edge
        #in between  only an edge if attached to an edge above thresh 2
        self.frame = cv2.Canny(self.frame, threshold1=100, threshold2=200)
        
        # src	input image.
        # dst	output image of the same size and the same number of channels as src .
        # ddepth	output image depth, see combinations; in the case of 8-bit input images it will result in truncated derivatives.
        # dx	order of the derivative x.
        # dy	order of the derivative y.
        # ksize	size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
        # scale	optional scale factor for the computed derivative values; by default, no scaling is applied (see getDerivKernels for details).
        # delta	optional delta value that is added to the results prior to storing them in dst.
        # borderType	pixel extrapolation method, see BorderTypes. BORDER_WRAP is not supported. 

        # sobelx = cv2.Sobel(self.frame, cv2.CV_64F, 1, 0, ksize=3)  # X direction
        # sobely = cv2.Sobel(self.frame, cv2.CV_64F, 0, 1, ksize=3)  # Y direction
        # self.frame = cv2.magnitude(sobelx, sobely)               # Combined magnitude
        

        #src	Source image.
        # dst	Destination image of the same size and the same number of channels as src .
        # ddepth	Desired depth of the destination image, see combinations.
        # ksize	Aperture size used to compute the second-derivative filters. See getDerivKernels for details. The size must be positive and odd.
        # scale	Optional scale factor for the computed Laplacian values. By default, no scaling is applied. See getDerivKernels for details.
        # delta	Optional delta value that is added to the results prior to storing them in dst .
        # borderType	Pixel extrapolation method, see BorderTypes. BORDER_WRAP is not supported. 
        # self.frame = cv2.Laplacian(self.frame, cv2.CV_64F)

        #HoughLine detection
        # horizontal_kernel = np.array([[-1, -1, -1],
        #                       [ 2,  2,  2],
        #                       [-1, -1, -1]])
        # self.frame = cv2.filter2D(self.frame, -1, horizontal_kernel)

        # self.frame = cv2.HoughLinesP(self.frame, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    def contours(self):
        # Find Contours
        contours, _ = cv2.findContours(self.frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw the contours on the original image
        cv2.drawContours(self.original, contours, -1, (0, 255, 0), 2)

        return contours

    def boundingBoxes(self, contours):
        obstacles = []
        # Draw bounding boxes
        for cnt in contours:
            # # Upright bounding boxes
            # poly = cv2.approxPolyDP(cnt, 3, True)
            # cnt = cv2.convexHull(cnt)
            # x,y,w,h = cv2.boundingRect(cnt)
            # if (w * h > 0):
                # cv2.rectangle(self.frame,(x,y),(x+w,y+h),(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)),2)
                # cv2.rectangle(self.frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

            # Rotated bounding boxes
            rect = cv2.minAreaRect(cnt)
            # Check if width * height > minArea
            if rect[1][0] * rect[1][1] > 250:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(self.original,[box],0,(0,0,255),2)
                obstacle = BoundedObstacle()
                obstacle.setRect(rect)
                obstacle.setCorners(box)
                obstacles.append(obstacle)
        
        return obstacles

    def featureDetection(self, obstacle):
        # Compute the bounding rectangle that encompasses the corners
        x, y, w, h = cv2.boundingRect(obstacle.corners)
    
        # Take cropped image and find features in it
        keypoints = self.fast.detect(self.frame, None)
        new_kp = []
        # only for visualization DELETE LATER
        for kp in keypoints:
            if (x <= kp.pt[0] <= x + w and y <= kp.pt[1] <= y + h):
                new_kp.append(kp)
        # self.original = cv2.drawKeypoints(self.original, new_kp, None, color=(255,255,0))
        # Convert keypoints to a list of coordinates (needed by calcOpticalFlowPyrLK)
        # np_points = np.array([[kp.pt] for kp in keypoints], dtype=np.float32)
        # Filter the keypoints and create the numpy array simultaneously
        np_points = np.array(
            [[kp.pt] for kp in keypoints if (x <= kp.pt[0] <= x + w and y <= kp.pt[1] <= y + h)],
            dtype=np.float32
        )

        # Create a mask image for drawing purposes
        # mask = np.zeros_like(original_gray)
        
        # return np_points, mask
        return TrackedObject(np_points)

    # mask needs to be initialized outside of function
    # def opticalFlow(self, p0, mask):
    def opticalFlow(self, obstacle):

        p0 = obstacle.points

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.frame, p0, None, **self.lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), [128, 128, 128], 2)
            self.original = cv2.circle(self.original, (int(a), int(b)), 2, [255, 0, 0], -1)
        # self.frame = cv2.add(self.frame, mask)

        # Now update the previous frame and previous points
        self.old_frame = self.frame.copy()
        obstacle.points = good_new.reshape(-1, 1, 2)

        # return mask

    def processImage(self, imgPath): # Returns list of BoundedObstacle objects
        contours = obstacles = []
        self.loadImage(imgPath)
        self.grayscale()
        self.blur()
        self.threshold()
        self.morphology()
        self.edgeDetection()
        # pipeline.displayImage('other')
        contours = self.contours()
        obstacles = self.boundingBoxes(contours)
        # self.displayImage('rgb')
        return obstacles
    
    def processVideoFrame(self): # Returns list of BoundedObstacle objects
        contours = obstacles = []
        self.loadFrame()
        self.grayscale()
        # self.blur()
        self.threshold()
        self.morphology()
        self.edgeDetection()
        # self.displayFrame('other')
        contours = self.contours()
        obstacles = self.boundingBoxes(contours)
        # self.displayFrame('rgb')
        return obstacles

    def displayImage(self, type):
        if (type == 'rgb'):
            cv2.imshow('Processed Image', self.original)
        else:
            cv2.imshow('Processed Image', self.frame)

        cv2.waitKey(0)

    def displayFrame(self, type):
        if (type == 'rgb'):
            cv2.rectangle(self.original, (200, 100), (450, 200), color=(255, 0, 0), thickness=5)
            cv2.imshow('Processed Image', self.original)
        else:
            cv2.imshow('Processed Image', self.frame)

if __name__ == '__main__':
    analysis = ImageAnalysis()
    obstacles = analysis.processImage('ComputerVision/testImages/img2.jpg')