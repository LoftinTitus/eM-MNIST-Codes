import pandas as pd
import math

# Enter your manual coordinates here as a list of tuples
manual_corners = [(391, 427), (1563, 427), (1563, 1586), (391, 1586)]

# Read coordinates
df = pd.read_csv("/Users/tyloftin/Library/CloudStorage/Box-Box/Titus/MNIST Data Set/Processed DaVis Data/020_Images/020_Images0001.csv", delimiter=';').values
xs = df[0]
ys = df[1]
csv_corners = [
    (xs.min(), ys.min()),
    (xs.min(), ys.max()),
    (xs.max(), ys.min()),
    (xs.max(), ys.max())
]

# Measure distances between corresponding corners
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

distances = [euclidean(manual_corners[i], csv_corners[i]) for i in range(4)]
print("Distances between corresponding corners:", distances)