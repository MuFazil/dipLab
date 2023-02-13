#CS20B1012
#Muhammad Fazil K

#Importing modules
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Loading the original image
f = plt.imread('lena.png')

# Input array
k=[5,10,20,30]


for n in k : 
    # Creating an empty array to store the noisy images
    fi = np.empty((n, f.shape[0], f.shape[1], f.shape[2]), dtype=np.float32)

    # Generating the noisy images
    for i in range(n):
        # Adding Gaussian noise with mean = 0 and variance = 1
        noise = np.random.normal(0, 1, f.shape)
        fi[i, :, :, :] = f + noise

    # Finding the average image (g)
    g = np.mean(fi, axis=0)

    # Displaying the original image, average image, and noisy images
    if n==5 :
        plt.figure(figsize=(10, 10))
        plt.subplot(3, 3, 1)
        plt.imshow(f)
        plt.title('Original Image')

        plt.subplot(3, 3, 2)
        plt.imshow(g)
        plt.title('Average Image')

        for i in range(n):
            plt.subplot(3, 3, i+3)
            plt.imshow(fi[i, :, :, :])
            plt.title(f'Noisy Image {i+1}')

    elif n==10 :
        plt.figure(figsize=(10, 10))
        plt.subplot(3, 5, 1)
        plt.imshow(f, cmap='gray')
        plt.title('Original Image')

        plt.subplot(3, 5, 2)
        plt.imshow(g, cmap='gray')
        plt.title('Average Image')

        for i in range(n):
            plt.subplot(3, 5, i+3)
            plt.imshow(fi[i, :, :], cmap='gray')
            plt.title(f'Noisy Image {i+1}')

    elif n==20 :
        
        plt.figure(figsize=(10, 10))
        plt.subplot(4, 6, 1)
        plt.imshow(f, cmap='gray')
        plt.title('Original Image')

        plt.subplot(4, 6, 2)
        plt.imshow(g, cmap='gray')
        plt.title('Average Image')

        for i in range(n):
            plt.subplot(4, 6, i+3)
            plt.imshow(fi[i, :, :], cmap='gray')
            plt.title(f'Noisy Image {i+1}')

    elif n==30 :
        plt.figure(figsize=(10, 10))
        plt.subplot(4, 8, 1)
        plt.imshow(f, cmap='gray')
        plt.title('Original Image')

        plt.subplot(4, 8, 2)
        plt.imshow(g, cmap='gray')
        plt.title('Average Image')


        for i in range(n):
            plt.subplot(4, 8, i+3)
            plt.imshow(fi[i, :, :], cmap='gray')
            plt.title(f'Noisy Image {i+1}')
    
    else : 
        print("Invalid number")

    plt.show()
