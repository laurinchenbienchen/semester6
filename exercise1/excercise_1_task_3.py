import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu, threshold_yen, threshold_li, threshold_mean, threshold_isodata, \
    threshold_triangle

BASE_DIRECTORY = '/Users/laurinerichlitzki/laurinesrepository/semester6/img'

image = io.imread(os.path.join(BASE_DIRECTORY, 'cells.png'))



#  thresholding mit vorgegebenen schwellwert T
def basic_thresholding(image, threshold):
    result = np.where(image >= threshold, 0, 255).astype(np.uint8)
    return result


# berechnung des schwellwertes mittels optimalen thresholding aus vl
def compute_threshold(image):
    T = np.mean(image)
    T_old = float('inf')

    while T_old != T:
        values1 = image[image >= T]
        values2 = image[image < T]


        mü1 = np.mean(values1)
        mü2 = np.mean(values2)
        T_old = T

        T = (mü1 + mü2) / 2

        return int(T)


# hilfsfunktion trheshold anwenden, zellen weiß
def aplly_thresholding(image, threshold):
    binary = np.where(image >= threshold, 0, 255).astype(np.uint8)
    return binary

if __name__ == '__main__':
# schwellwerte berechnen
    T_optimal = compute_threshold(image)
    print(f"interativer schwellwert {T_optimal}")
# interativer schwellwert 186
    T_otsu = threshold_otsu(image)
    print(f"otsu schwellwert {T_otsu}")
# otsu schwellwert 186
    T_yen = threshold_yen(image)
    print(f"yen schwellwert {T_yen}")
# yen schwellwert 118

    T_li = threshold_li(image)
    print(f"li-schwellwert {T_li:.4f}")
# li-schwellwert 184.4199
    T_mean = threshold_mean(image)
    print(f"mean-schwellwert {T_mean:.4f}")
# mean-schwellwert 196.6983
    T_triangle = threshold_triangle(image)
    print(f"triangle-schwellwert {T_triangle:.4f}")
# triangle-schwellwert 205.0000
    T_isodata = threshold_isodata(image)
    print(f"isodata-schwellwert {T_isodata:.4f}")
# isodata-schwellwert 186.0000

    binary_iterative = aplly_thresholding(image, T_optimal)
    binary_otsu = aplly_thresholding(image, T_otsu)
    binary_yen = aplly_thresholding(image, T_yen)

    binary_li = aplly_thresholding(image, T_li)
    binary_mean = aplly_thresholding(image, T_mean)
    binary_triangle = aplly_thresholding(image, T_triangle)
    binary_isodata = aplly_thresholding(image, T_isodata)

    titles = ['Original', f'Optimsl (T={T_optimal})', f'Otsu (T={T_otsu})', f'Yen (T={T_yen})', f'Li (T={T_li})', f'Mean (T={T_mean})', f'Triangle (T={T_triangle})',f'Isodata (T={T_isodata})',]
    images = [image, binary_iterative, binary_otsu, binary_yen, binary_li, binary_mean,binary_triangle, binary_isodata]

    plt.figure(figsize=(16, 6))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()