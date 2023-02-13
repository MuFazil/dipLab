import cv2
import numpy as np

# Load the image
img = cv2.imread("PISA.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define a function for bilinear interpolation


def biLinearInterpolation(x, y, img):
    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    y1 = int(np.floor(y))
    y2 = int(np.ceil(y))

    f11 = img[y1][x1]
    f12 = img[y2][x1]
    f21 = img[y1][x2]
    f22 = img[y2][x2]

    f = (x2 - x) * ((y2 - y) * f11 + (y - y1) * f12) + \
        (x - x1) * ((y2 - y) * f21 + (y - y1) * f22)

    return f

# Find the angle of inclination


def findAngleofInclination(img):
    H, W = img.shape[:2]
    angles = np.arange(-45, 46)

    scores = []
    for angle in angles:
        M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1)
        rotated = cv2.warpAffine(
            img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        total_sum = 0
        for y in range(H):
            for x in range(W):
                if x == 0 or x == W-1 or y == 0 or y == H-1:
                    total_sum += rotated[y][x]
                else:
                    total_sum += biLinearInterpolation(x, y, rotated)
        scores.append(total_sum)

    angle_of_inclination = angles[np.argmax(scores)]
    return angle_of_inclination


angle = findAngleofInclination(gray)
print("The angle of inclination :", angle, "degrees")
