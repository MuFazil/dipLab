# CS20B1012
# Muhammad Fazil K

# Importing modules
import cv2
import numpy as np

# Reading the input images
img = cv2.imread('pout-dark.jpg', 0)
ref_img = cv2.imread('pout-bright.jpg', 0)

# In-Built : Histogram Equalization
equalized_img = cv2.equalizeHist(img)

# Computing the histograms of the input and reference images
img_hist, _ = np.histogram(img.flatten(), 256, [0, 256])
ref_hist, _ = np.histogram(ref_img.flatten(), 256, [0, 256])

# Computing the cumulative histograms of the input and reference images
img_cdf = img_hist.cumsum()
img_cdf_normalized = img_cdf * img_hist.max() / img_cdf.max()

ref_cdf = ref_hist.cumsum()
ref_cdf_normalized = ref_cdf * img_hist.max() / ref_cdf.max()

# Computing the lookup table for the histogram matching
lookup_table = np.interp(
    img_cdf_normalized, ref_cdf_normalized, range(256)).astype('uint8')

# Applying the histogram matching to the input image
matched_img = cv2.LUT(img, lookup_table)


# User Defined : Equalization
def histoEq_UD(image):
    # Computing histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Computing cumulative distribution function
    cdf = hist.cumsum()

    # Normalizing CDF
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Generating equalized image
    equalized = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)

    return equalized.astype(np.uint8)


# User defined : histogram matching

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


# Performing histogram equalization
eq_img_UD = histoEq_UD(img)

# Performing histogram matching (specification)
mat_img_UD = histoMat_UD(img, ref_img)

# Displaying the images
cv2.imshow('Input', img)
cv2.imshow('IB_Matched', matched_img)
cv2.imshow('IB_Equalized', equalized_img)
cv2.imshow('UD_Matched', mat_img_UD)
cv2.imshow('UD_Equalized', eq_img_UD)
cv2.waitKey(0)
cv2.destroyAllWindows()
