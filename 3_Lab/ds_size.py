# CS20B1012
# Muhammad Fazil K

# Importing necessary modules
import cv2
import matplotlib.pyplot as plt

# Loading the Lena image
img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

# Creating a list of down-sampled image shapes
imgSizes = [(128, 128), (64, 64), (32, 32), (16, 16)]

# Down-sampling the image using cv2.resize()
dsImages = [cv2.resize(img, shape) for shape in imgSizes]

# Displaying
fig, axs = plt.subplots(1, len(dsImages) + 1, figsize=(15, 15))
axs[0].imshow(img, cmap="gray")
axs[0].set_title("Original")
for i, down_img in enumerate(dsImages):
    axs[i + 1].imshow(down_img, cmap="gray")
    axs[i + 1].set_title(f"Down-sampled ({imgSizes[i][0]}x{imgSizes[i][1]})")
plt.show()
