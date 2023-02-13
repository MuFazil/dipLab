# CS20B1012
# Muhammad Fazil K

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Lena.png", cv2.IMREAD_GRAYSCALE)

# Defining the intensity ranges
intRanges = [(0, 255), (0, 128), (0, 64), (0, 32),
             (0, 16), (0, 8), (0, 4), (0, 2)]

# Down-sampling the image for each range
dsImages = []
for r in intRanges:
    img_temp = img.copy()
    img_temp = cv2.normalize(
        img_temp, None, alpha=r[0], beta=r[1], norm_type=cv2.NORM_MINMAX)
    dsImages.append(img_temp)

# Displaying
plt.figure(figsize=(12, 12))
plt.subplot(3, 3, 1)
plt.title("Original")
plt.imshow(img, cmap="gray")

for i in range(len(dsImages)):
    plt.subplot(3, 3, i + 2)
    plt.title("Range: {} - {}".format(intRanges[i][0], intRanges[i][1]))
    plt.imshow(dsImages[i], cmap="gray")

plt.show()
