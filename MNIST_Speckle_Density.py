import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology, draw, exposure
import numpy as np
from skimage.filters import gaussian
import pandas as pd
import cv2

# Load image
image_path = "/Users/tyloftin/Library/CloudStorage/Box-Box/Titus/MNIST Data Set/Sample Images/000Images/000_1_00000.jpg"
image = io.imread(image_path)

# Make grayscale
gray = color.rgb2gray(image) if image.ndim == 3 else image

# Calculate mean image gradient using OpenCV
gray_cv = (gray * 255).astype(np.uint8)
grad_x = cv2.Sobel(gray_cv, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray_cv, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
mean_gradient = np.mean(grad_mag)
print(f"Mean image gradient: {mean_gradient:.2f}")

# Get rid of uneven illumination before gamma correction
background = gaussian(gray, sigma=50)
gray = gray - background
gray = np.clip(gray, 0, 1)

# Apply gamma correction for contrast enhancement
gray = exposure.adjust_gamma(gray, gamma=3)

# Smooth
gray = gaussian(gray, sigma=1)

# Threshhold
thresh = filters.threshold_otsu(gray)
binary = gray > thresh

# detech ROI
ys, xs = np.where(binary > 0)
x_min, x_max = np.min(xs), np.max(xs)
y_min, y_max = np.min(ys), np.max(ys)

#  constrain bounds to avoid handles
y_min = max(y_min, 400)
y_max = min(y_max, 1600)
x_min = max(x_min, 200)
x_max = min(x_max, 1600)

# Create a mask of the detected sample ROI
roi_mask = np.zeros_like(binary, dtype=bool)
roi_mask[y_min:y_max, x_min:x_max] = True

# roi mask
binary_roi = binary & roi_mask
labels = measure.label(binary_roi, connectivity=2)
props = measure.regionprops(labels)

# calculate the area
sample_area = np.sum(binary_roi)


speckle_areas = [prop.area for prop in props]
speckle_count = len(speckle_areas)
avg_speckle_size = np.mean(speckle_areas)
std_speckle_size = np.std(speckle_areas)
speckle_density = speckle_count / 160 # area in mm^2

print(f"Speckle count: {speckle_count}")
print(f"Average speckle size: {avg_speckle_size:.2f} pixels")
print(f"Speckle size std dev: {std_speckle_size:.2f} pixels")
print(f"Speckle density: {speckle_density:.6f} per mm^2")

# Plot grayscale image with ROI and speckles
plt.figure(figsize=(6,6))
plt.imshow(gray, cmap='gray')

# Draw detected ROI border
rect_y, rect_x = draw.rectangle_perimeter(start=(y_min, x_min), end=(y_max-1, x_max-1))
plt.plot(rect_x, rect_y, color='blue', linewidth=2, label='Sample ROI')

# Plot speckles
for prop in props:
    y, x = prop.centroid
    plt.scatter(x, y, c='blue', alpha=0.6)

plt.title("Speckles")
plt.axis('off')
plt.legend()
plt.show()

