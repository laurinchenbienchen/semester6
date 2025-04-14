# image segmentation

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage
from skimage import color, exposure, filters, morphology, measure
from skimage.util import img_as_ubyte
from exercise1 import excercise_1_task_3



BASE_DIRECTORY = '/Users/laurinerichlitzki/laurinesrepository/semester6/img'

image_cells = io.imread(os.path.join(BASE_DIRECTORY, 'fluo_nuclei_bbc.png'))
image_object1 = io.imread(os.path.join(BASE_DIRECTORY, 'objects_normal_1.png'))
image_object2 = io.imread(os.path.join(BASE_DIRECTORY, 'objects_normal_2.png'))
image_object3 = io.imread(os.path.join(BASE_DIRECTORY, 'objects_normal_3.png'))
image_wdg = io.imread(os.path.join(BASE_DIRECTORY, 'wdg3.png'))

# subimages erstellen um hintergrund auszugleichen
# thresholding auf subimages anwenden
# global thresholding

def global_threshilding(image):
    block_size = 137
    local_thresh = skimage.filters.threshold_local(image, block_size, offset=10)
    binary_local = image > local_thresh
    binary_local = binary_local.astype(np.uint8)

    return binary_local


def local_threshilding_objects(image, block_size):
    print(type(image))
    print(image.shape)
    print(image.ndim)
    h,w = image.shape

    image = skimage.filters.gaussian(image, sigma=10) # bei den kreisen relevant
    image = skimage.exposure.equalize_adapthist(image)
    image = skimage.filters.median(image)

    result = np.zeros_like(image, dtype=np.uint8)

    for i in range (0,h,block_size):
        for j in range (0,w,block_size):
            y_end = min(i + block_size, h)
            x_end = min(j + block_size, w)

            block = image[i:y_end, j:x_end]

            threshold = skimage.filters.threshold_otsu(block)

            binary_block = (block > threshold).astype(np.uint8) * 255
            result[i:y_end, j:x_end] = binary_block

    return result

def local_thresholding_wdg(image, block_size):
    image = skimage.color.rgb2gray(image)
    print(type(image))
    print(image.shape)
    print(image.ndim)
    h, w = image.shape

    image = skimage.exposure.equalize_adapthist(image)
    #image = skimage.filters.gaussian(image, sigma=1.8)
    image = skimage.filters.median(image)

    result = np.zeros_like(image, dtype=np.uint8)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            y_end = min(i + block_size, h)
            x_end = min(j + block_size, w)

            block = image[i:y_end, j:x_end]  #

            threshold = skimage.filters.threshold_otsu(block)

            binary_block = (block > threshold).astype(np.uint8) * 255
            result[i:y_end, j:x_end] = binary_block

    return result


if __name__ == '__main__':
    T_optimal = excercise_1_task_3.compute_threshold(image_cells)
    print(f"interativer schwellwert {T_optimal}")

    binary_iterative = excercise_1_task_3.aplly_thresholding(image_cells, T_optimal)
    plt.imshow(binary_iterative, cmap='gray')
    plt.show()

    glo_thres = local_threshilding_objects(image_object1, block_size=600)
    plt.imshow(glo_thres, cmap='gray')
    plt.show()

    glo_thres = local_threshilding_objects(image_object2, block_size=600)
    plt.imshow(glo_thres, cmap='gray')
    plt.show()

    glo_thres = local_threshilding_objects(image_object3, block_size=600)
    plt.imshow(glo_thres, cmap='gray')
    plt.show()

    binary_image = local_thresholding_wdg(image_wdg, block_size=200)
    plt.imshow(binary_image, cmap='gray')
    plt.show()
