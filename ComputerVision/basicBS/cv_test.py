import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "/home/dorian/Repos/SeniorDesign/ComputerVisionTesting/ComputerVision/testImages/img1.jpg"
image = cv2.imread(image_path)
original_image = image.copy()

# Step 1: Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# Step 2: Apply Bilateral Filter or Thresholding
# Option A: Use bilateral filter to reduce noise while keeping edges sharp
filtered = cv2.bilateralFilter(gray, d=10   , sigmaColor=50, sigmaSpace=100)

# Option B: Use thresholding instead of bilateral filter (uncomment if needed)
# _, filtered = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Step 3: Keypoint Detection with ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(filtered, None)

# Draw keypoints on the original image
image_with_keypoints = cv2.drawKeypoints(original_image, keypoints, None, color=(0, 255, 0), flags=0)

# Step 4: Edge Detection using Cann2
edges = cv2.Canny(filtered, threshold1=10, threshold2=200)

# Step 5: Find Contours and Draw Rectangles
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Convert images to RGB format for matplotlib (since OpenCV uses BGR)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_with_keypoints_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)

# Display the images with matplotlib
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Image Processing Steps", fontsize=16)

# Show each step in a subplot
axes[0, 0].imshow(original_image_rgb)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Grayscale Image")
axes[0, 1].axis("off")

axes[0, 2].imshow(filtered, cmap="gray")
axes[0, 2].set_title("Filtered/Thresholded Image")
axes[0, 2].axis("off")

axes[1, 0].imshow(image_with_keypoints_rgb)
axes[1, 0].set_title("Keypoints")
axes[1, 0].axis("off")

axes[1, 1].imshow(edges, cmap="gray")
axes[1, 1].set_title("Edges")
axes[1, 1].axis("off")

axes[1, 2].imshow(image_rgb)
axes[1, 2].set_title("Contours with Rectangles")
axes[1, 2].axis("off")

# Adjust layout and show the plot
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
