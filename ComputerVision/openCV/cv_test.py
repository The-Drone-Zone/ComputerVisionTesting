import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "/home/dorian/Repos/SeniorDesign/ComputerVisionTesting/ComputerVision/testImages/img2.jpg"
image = cv2.imread(image_path)
original_image = image.copy()

# Step 1: Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Bilateral Filter or Thresholding
# Option A: Use bilateral filter to reduce noise while keeping edges sharp
filtered = cv2.bilateralFilter(gray, d=15, sigmaColor=25, sigmaSpace=50)

# Option B: Use thresholding instead of bilateral filter (uncomment if needed)
# _, filtered = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Step 3: Edge Detection using Canny
edges = cv2.Canny(filtered, threshold1=200, threshold2=200)

# Step 4: Dilation to Connect Broken Edges
kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# Step 5: Apply Morphological Operations to Refine Edges
# Closing - fills small holes in edges
closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)

# Gradient - highlights the edges
gradient_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_GRADIENT, kernel)

# Step 6: Find Contours and Draw Rectangles on the Closed Edges
contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Convert images to RGB format for matplotlib (since OpenCV uses BGR)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the images with matplotlib
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
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

axes[1, 0].imshow(edges, cmap="gray")
axes[1, 0].set_title("Edges")
axes[1, 0].axis("off")

axes[1, 1].imshow(dilated_edges, cmap="gray")
axes[1, 1].set_title("Dilated Edges")
axes[1, 1].axis("off")

axes[1, 2].imshow(closed_edges, cmap="gray")
axes[1, 2].set_title("Closed Edges for Contour Detection")
axes[1, 2].axis("off")

axes[2, 0].imshow(gradient_edges, cmap="gray")
axes[2, 0].set_title("Gradient Edges")
axes[2, 0].axis("off")

axes[2, 1].imshow(image_rgb)
axes[2, 1].set_title("Contours with Rectangles")
axes[2, 1].axis("off")

# Adjust layout and show the plot
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
