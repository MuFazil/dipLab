
import cv2
import numpy as np

# Loading the image
img = cv2.imread("Lena.png")

# Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Scaling by factor of 1
scaledUser1 = gray.copy()

# Scaling by factor of 2
scaledUser2 = np.zeros((gray.shape[0]*2, gray.shape[1]*2),dtype=np.uint8)
for i in range(gray.shape[0]*2):
    for j in range(gray.shape[1]*2):
        x, y = i / 2, j / 2
        x1, x2, y1, y2 = int(x), int(x) + 1, int(y), int(y) + 1

        a = x - x1
        b = y - y1

        if x1 >= gray.shape[0] - 1:
            x1 -= 1
        if x2 >= gray.shape[0] - 1:
            x2 -= 1
        if y1 >= gray.shape[1] - 1:
            y1 -= 1
        if y2 >= gray.shape[1] - 1:
            y2 -= 1

        scaledUser2[i, j] = (1 - a) * (1 - b) * gray[x1, y1] + a * (1 - b) * gray[x2, y1] + (1 - a) * b * gray[
            x1, y2] + a * b * gray[x2, y2]

# Scaling by factor of 0.5
scaledUser05 = np.zeros((gray.shape[0]//2, gray.shape[1]//2),dtype=np.uint8)
for i in range(gray.shape[0]//2):
    for j in range(gray.shape[1]//2):
        x, y = i * 2, j * 2
        x1, x2, y1, y2 = int(x), int(x) + 1, int(y), int(y) + 1

        a = x - x1
        b = y - y1

        if x1 >= gray.shape[0] - 1:
            x1 -= 1
        if x2 >= gray.shape[0] - 1:
            x2 -= 1
        if y1 >= gray.shape[1] - 1:
            y1 -= 1
        if y2 >= gray.shape[1] - 1:
            y2 -= 1

        scaledUser05[i, j] = (1 - a) * (1 - b) * gray[x1, y1] + a * (1 - b) * gray[x2, y1] + (1 - a) * b * gray[
            x1, y2] + a * b * gray[x2, y2]

# Using in-built functions

# Scaling by factor of 1
scaledBuiltin1 = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

# Scaling by factor of 2
scaledBuiltin2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Scaling by factor of 0.5
scaledBuiltin05 = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# Display the original and scaled images
cv2.imshow("Original", gray)
cv2.imshow("Scaled by factor of 1", scaledUser1)
cv2.imshow("Scaled by factor of 2", scaledUser2)
cv2.imshow("Scaled by factor of 0.5", scaledUser05)

# Display the images scaled using built-in functions
cv2.imshow("Scaled by factor of 1 (Built-in Function)", scaledBuiltin1)
cv2.imshow("Scaled by factor of 2 (Built-in Function)", scaledBuiltin2)
cv2.imshow("Scaled by factor of 0.5 (Built-in Function)", scaledBuiltin05)

cv2.waitKey(0)
cv2.destroyAllWindows()