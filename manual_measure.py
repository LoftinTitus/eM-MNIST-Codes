import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import cdist
import cv2
import matplotlib.pyplot as plt

# Enter your manual coordinates here as a list of tuples
manual_corners =      [(337, 386), (1463, 386), (1463, 1509), (337, 1509)]
# Read coordinates from CSV (similar to MNIST_Mask_Measure.py)
csv_path = "/Users/tyloftin/Library/CloudStorage/Box-Box/Titus/MNIST Data Set/Processed DaVis Data/006_Images/006_Images0001.csv"
M_data = pd.read_csv(csv_path, delimiter=';').values
x, y = M_data[:, 0], M_data[:, 1]

# Flip y coordinates to match image origin (top-left) - assuming typical image height
img_height = 1652  # Adjust this based on your actual image height
y = img_height - y

# Create corners from the bounding box of all CSV points
csv_corners = [
    (np.min(x), np.min(y)),
    (np.max(x), np.min(y)),
    (np.max(x), np.max(y)),
    (np.min(x), np.max(y))
]

# Build boundaries for both sets
nEdgePts = 200
def build_boundary(corners):
    corners = np.array(corners + [corners[0]])  # close polygon
    boundary = []
    for k in range(len(corners) - 1):
        x_line = np.linspace(corners[k, 0], corners[k + 1, 0], nEdgePts)
        y_line = np.linspace(corners[k, 1], corners[k + 1, 1], nEdgePts)
        boundary.extend(np.column_stack((x_line, y_line)))
    return np.array(boundary)

B_manual = build_boundary(manual_corners)
B_csv = build_boundary(csv_corners)

# Find distances
D = cdist(B_manual, B_csv)
minDist = np.min(D, axis=1)
meanDist = np.mean(minDist)
maxDist = np.max(minDist)

# Create binary masks for IoU
img_shape = (int(max(max(B_manual[:,1]), max(B_csv[:,1]))+10), int(max(max(B_manual[:,0]), max(B_csv[:,0]))+10))
M = np.zeros(img_shape, dtype=np.uint8)
S = np.zeros_like(M)
cv2.fillPoly(M, [np.int32(B_manual)], 1)
cv2.fillPoly(S, [np.int32(B_csv)], 1)
intersection = cv2.bitwise_and(M, S)
union = cv2.bitwise_or(M, S)
iou = 100 * np.sum(intersection) / np.sum(union)

print("Mean Distance:", meanDist)
print("Max Distance:", maxDist)
print("Fit % (IoU):", iou)

# Visual
plt.figure(figsize=(8, 8))
plt.plot(B_manual[:, 0], B_manual[:, 1], 'r-', label='Manual Border')
plt.plot(B_csv[:, 0], B_csv[:, 1], 'g-', label='CSV Border')
plt.title('Manual vs CSV Borders')
plt.legend()
plt.gca().invert_yaxis()
plt.show()