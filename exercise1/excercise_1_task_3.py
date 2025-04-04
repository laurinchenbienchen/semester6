import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

BASE_DIRECTORY = '/Users/laurinerichlitzki/laurinesrepository/semester6/img'

image = io.imread(os.path.join(BASE_DIRECTORY, 'cells.png'))

#  thresholding mit vorgegebenen schwellwert T + testprogramm

def basic_thresholding(image,threshold):
    height = image.shape[0]
    width = image.shape[1]
    for i in range(height):
        for j in range(width):
            if image[i,j] >= threshold:
                image[i,j] = 255
            else:
                image[i,j] = 0
    # values = image [image < threshold]
    return image

def compute_threshold(image, threshold):


image_thresh = basic_thresholding(image,187)
plt.imshow(image_thresh, cmap='gray')
plt.show()