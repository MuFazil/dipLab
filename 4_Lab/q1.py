import cv2
import numpy as np

# Load the image
img = cv2.imread("D:\SEM6\dipLab\4_Lab\Picture_5.png")

# Convert the image to grayscale for processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get the height and width of the image
h, w = gray.shape

# Define the center of the image
center = (w // 2, h // 2)

# Define the scale factor for resizing the image
scale = 0.5

# Resize the image for faster processing
gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)

# Perform Gaussian Blurring to reduce noise in the image
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny Edge Detection to get the edges in the image
edges = cv2.Canny(gray, 50, 150)

# Perform Hough Line Transform to get the lines in the image
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                        minLineLength=100, maxLineGap=10)

# Initialize the angle of inclination to 0
angle = 0

print(type(lines))
# Loop through the lines to find the line with the maximum slope
for line in lines:
    x1, y1, x2, y2 = line[0]
    slope = (y2 - y1) / (x2 - x1)
    if abs(slope) > abs(angle):
        angle = slope

# Convert the slope to degrees
angle = np.arctan(angle) * 180 / np.pi

# Perform rotation to correct the inclination of the tower
rotated = np.array(gray)
if angle < 0:
    angle = -90 - angle
else:
    angle = 90 - angle
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(
    gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# Perform bilinear interpolation to resize the image back to its original size
rotated = cv2.resize(rotated, (0, 0), fx=1/scale, fy=1 /
                     scale, interpolation=cv2.INTER_LINEAR)

# Save the rotated image
cv2.imwrite("rotated_pisa.jpg", rotated)

# Print the angle of inclination
print("Angle of inclination: {:.2f} degrees".format(angle))
