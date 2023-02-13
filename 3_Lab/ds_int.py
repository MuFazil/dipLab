import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Lena image
img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

# Create a list of intensity ranges
intensity_ranges = [255, 128, 64, 32, 16, 8, 4, 2]

# Down-sample the image using intensity range
downsampled_imgs = [np.uint8(np.round(img / intensity_range) * intensity_range)
                    for intensity_range in intensity_ranges]

# Display the down-sampled images
fig, axs = plt.subplots(2, 4, figsize=(15, 15))
for i, down_img in enumerate(downsampled_imgs):
    ax = axs.flat[i]
    ax.imshow(down_img, cmap="gray")
    ax.set_title(f"Intensity range 0-{intensity_ranges[7-i]}")
plt.show()
