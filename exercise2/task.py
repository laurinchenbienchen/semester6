# image segmentation

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage
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


def local_threshilding(image, block_size):
    #image = skimage.color.rgb2gray(image)  # an und aus schalten von wdg und kreis
    print(type(image))
    print(image.shape)
    print(image.ndim)
    h,w = image.shape

    image = skimage.filters.gaussian(image, sigma=15) # bei den kreisen relevant
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


if __name__ == '__main__':
    T_optimal = excercise_1_task_3.compute_threshold(image_cells)
    print(f"interativer schwellwert {T_optimal}")

    # binary_iterative = excercise_1_task_3.aplly_thresholding(image_cells, T_optimal)
    # plt.imshow(binary_iterative, cmap='gray')
    # plt.show()

    # glo_thres = local_threshilding(image_object1, block_size=250)
    # plt.imshow(glo_thres, cmap='gray')
    # plt.show()

    binary_image = local_threshilding(image_object1, block_size=300)
    plt.imshow(binary_image, cmap='gray')
    plt.show()
