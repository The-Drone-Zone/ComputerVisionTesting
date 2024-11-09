import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def Harris(filePath):
    img = cv.imread(filePath)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    
    cv.imshow('dst',img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def ShiTomasi(filePath):
    img = cv.imread(filePath)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)
    
    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),3,255,-1)
    
    plt.imshow(img),plt.show()

def sift(filePath):
    img = cv.imread(filePath)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    
    img=cv.drawKeypoints(gray,kp,img)
    
    cv.imshow('SIFT', img)
    cv.waitKey(0)

def surf(filePath):
    img = cv.imread(filePath, cv.IMREAD_GRAYSCALE)

    # Create SURF object. You can specify params here or later.
    # Here I set Hessian Threshold to 400, should be between 300-500
    surf = cv.xfeatures2d.SURF_create(400)
    # Set to U-SURF since we don't care about direction
    surf.setUpright(True)

    # Find keypoints and descriptors directly
    kp = surf.detect(img,None)

    img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
    plt.imshow(img2),plt.show()


def fast(filePath):
    img = cv.imread(filePath, cv.IMREAD_GRAYSCALE) # `<opencv_root>/samples/data/blox.jpg`
 
    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()
    
    # find and draw the keypoints
    kp = fast.detect(img,None)
    img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
    
    # Print all default params
    print( "Threshold: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast.getType()) )
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
    
    cv.imshow('fast true', img2)
    
    # Disable nonmaxSuppression (MORE FEATURES)
    # fast.setNonmaxSuppression(0)
    # kp = fast.detect(img, None)
    
    # print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
    
    # img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
    
    # cv.imshow('fast false', img3)

    cv.waitKey(0)

def brief(filePath):
    img = cv.imread(filePath, cv.IMREAD_GRAYSCALE)
 
    # Initiate FAST detector
    star = cv.xfeatures2d.StarDetector_create()
    
    # Initiate BRIEF extractor
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    
    # find the keypoints with STAR
    kp = star.detect(img,None)
    
    # compute the descriptors with BRIEF
    kp, des = brief.compute(img, kp)
    
    print( brief.descriptorSize() )
    print( des.shape )

    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()

def orb(filePath):
    img = cv.imread(filePath, cv.IMREAD_GRAYSCALE)
 
    # Initiate ORB detector
    orb = cv.ORB_create()
    
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    
    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()

if __name__ == '__main__':
    filePath = 'ComputerVision/testImages/img2.jpg'
    # Harris(filePath) # BAD, NO POWER LINES
    # ShiTomasi(filePath) # TERRIBLE, NO ANYTHING
    sift(filePath) # DECENT, SOME POWER LINES DETECTED
    # surf(filePath) THIS IS PATENTED SO CANT WORK
    # fast(filePath) # GOOD, TONS OF FEATURES DETECTED
    # brief(filePath) # BAD, NO POWER LINES
    # orb(filePath) # TERRIBLE, , NO POWER LINES
