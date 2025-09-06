import numpy as np
import cv2
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Load the sample image
I_3D = cv2.imread("/Users/tyloftin/Library/CloudStorage/Box-Box/Titus/MNIST Data Set/Sample Images/020Images/020_1_00000.jpg")

# Convert to grayscale
if I_3D.shape[2] == 3:
    I_gray = cv2.cvtColor(I_3D, cv2.COLOR_BGR2GRAY)
else:
    I_gray = I_3D

# Apply gamma correction
# Normalize to [0,1], apply gamma, then scale back to [0,255]
gamma = 2
I_gray = np.power(I_gray / 255.0, gamma) * 255.0
I_gray = I_gray.astype(np.uint8)

img_height = I_gray.shape[0]

# Binarize image
_, I = cv2.threshold(I_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
I = I // 255  

# Mask data is from a csv
M_data = pd.read_csv("/Users/tyloftin/Library/CloudStorage/Box-Box/Titus/MNIST Data Set/Processed DaVis Data/020_Images/020_Images0001.csv", delimiter=';').values
x, y = M_data[:, 0], M_data[:, 1]

# Flip y coordinates to match image origin (top-left)
y = img_height - y

mask_coords = np.array([
    [np.min(x), np.min(y)],
    [np.max(x), np.min(y)],
    [np.max(x), np.max(y)],
    [np.min(x), np.max(y)],
    [np.min(x), np.min(y)]
])

# Build boundary for mask rectangle
nEdgePts = 200
B_mask = []
for k in range(mask_coords.shape[0] - 1):
    x_line = np.linspace(mask_coords[k, 0], mask_coords[k + 1, 0], nEdgePts)
    y_line = np.linspace(mask_coords[k, 1], mask_coords[k + 1, 1], nEdgePts)
    B_mask.extend(np.column_stack((x_line, y_line)))
B_mask = np.array(B_mask)

# Find borders of the sample in the real image
ys, xs = np.where(I > 0)
x_min, x_max = np.min(xs), np.max(xs)
y_min, y_max = np.min(ys), np.max(ys)

# Constrain to be within bounds so the code isnt picking up the handles
y_min = max(y_min, 375)  
y_max = min(y_max, 1550) 

sample_coords = np.array([
    [x_min, y_min],
    [x_max, y_min],
    [x_max, y_max],
    [x_min, y_max],
    [x_min, y_min]
])

# Boundary for the sample box
B_real = []
for k in range(sample_coords.shape[0] - 1):
    x_line = np.linspace(sample_coords[k, 0], sample_coords[k + 1, 0], nEdgePts)
    y_line = np.linspace(sample_coords[k, 1], sample_coords[k + 1, 1], nEdgePts)
    B_real.extend(np.column_stack((x_line, y_line)))
B_real = np.array(B_real)

# Find distances
D = cdist(B_mask, B_real)
minDist = np.min(D, axis=1)
meanDist = np.mean(minDist)
maxDist = np.max(minDist)

# make the mask binary
BW = I.astype(bool)

# Fill mask rectangle for IoU
M = np.zeros_like(BW, dtype=np.uint8)
cv2.fillPoly(M, [np.int32(B_mask)], 1)

# Fill sample rectangle
S = np.zeros_like(M, dtype=np.uint8)
cv2.fillPoly(S, [np.int32(B_real)], 1)

# IoU calc
intersection = cv2.bitwise_and(M, S)
union = cv2.bitwise_or(M, S)
iou = 100 * np.sum(intersection) / np.sum(union)

print("Mean Distance:", meanDist)
print("Max Distance:", maxDist)
print("Fit %:", iou)

# Visual
plt.figure(figsize=(8, 8))
plt.imshow(I_gray, cmap='gray')
plt.plot(B_mask[:, 0], B_mask[:, 1], 'r-', label='Mask Border')
plt.plot(B_real[:, 0], B_real[:, 1], 'g-', label='Sample Border')
plt.title('Mask vs Sample Borders')
plt.legend()
plt.show()

