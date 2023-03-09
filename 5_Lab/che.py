import cv2
import numpy as np

# Define a function to perform histogram equalization


def histoEq_UD(image):
    # Compute histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Compute cumulative distribution function
    cdf = hist.cumsum()

    # Normalize CDF
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Generate equalized image
    equalized = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)

    return equalized.astype(np.uint8)


# Read the input image
img = cv2.imread('pout-dark.jpg', 0)


# Define a function to perform histogram matching


def histoMat_UD(img, ref_img):
    img_hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref_img.flatten(), 256, [0, 256])
    img_cdf = img_hist.cumsum()
    img_cdf_normalized = img_cdf * img_hist.max() / img_cdf.max()
    ref_cdf = ref_hist.cumsum()
    ref_cdf_normalized = ref_cdf * img_hist.max() / ref_cdf.max()
    lookup_table = np.interp(
        img_cdf_normalized, ref_cdf_normalized, range(256)).astype('uint8')
    mat_img_UD = cv2.LUT(img, lookup_table)
    return mat_img_UD.astype('uint8')


# Read the input images
img = cv2.imread('pout-dark.jpg', 0)
ref_img = cv2.imread('pout-bright.jpg', 0)

# Perform histogram equalization
eq_img_UD = histoEq_UD(img)

# Perform histogram matching (specification)
mat_img_UD = histoMat_UD(img, ref_img)

# Display the images

cv2.imshow('UD_Equalized', eq_img_UD)
cv2.imshow('UD_Matched', mat_img_UD)

cv2.waitKey(0)
cv2.destroyAllWindows()
