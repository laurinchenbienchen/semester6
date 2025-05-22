import numpy as np
from skimage import io
import os


BASE_DIRECTORY = '/Users/laurinerichlitzki/laurinesrepository/semester6/img'


# teilaufgabe a vergleich segmentierungs ergebnisse mit referenz segmentierung, berechnung mutual overlap metric

def calculate_area(img):
    # alle weißen pixel von interesse
    area = np.sum(img == 1)
    return area


def mutual_overlap(img1, img2):
    m1 = img1 == 1
    m2 = img2 == 1
    m_o = np.sum(np.logical_and(m1, m2))
    return m_o


def mutual_overlap_metric(img1, img2):
    m_o = mutual_overlap(img1, img2)
    area1 = calculate_area(img1)
    area2 = calculate_area(img2)

    m_mo = (2 * m_o) / (area1 + area2)
    return m_mo

# teilaufgabe b, segmentierungsverfahren einfaches thresholding, welcher treshold ist am besten

def basic_thresholding(img, threshold):
    img = img.copy()
    img[img <= threshold] = 0
    img[img > threshold] = 1
    return img

def find_best_threshold(original, reference):
    list_m_o=[]
    for i in range (0,225):
        threshold_img=basic_thresholding(original,i)
        mom= mutual_overlap_metric(threshold_img, reference)
        list_m_o.append([mom,i])
    return list_m_o


if __name__ == '__main__':
    method_1 = io.imread(os.path.join(BASE_DIRECTORY, 'IMG_2062_s_method1.png'), as_gray=True)
    method_2 = io.imread(os.path.join(BASE_DIRECTORY, 'IMG_2062_s_method2.png'), as_gray=True)
    reference = io.imread(os.path.join(BASE_DIRECTORY, 'IMG_2062_s_reference.png'), as_gray=True)
    original = io.imread(os.path.join(BASE_DIRECTORY, 'IMG_2062_s.png'))

    # teilaufgabe a
    result_method_1 = mutual_overlap_metric(method_1, reference)
    result_method_2 = mutual_overlap_metric(method_2, reference)


    print('result m_mo method 1:', result_method_1)
    print('result m_mo method 2:', result_method_2)

    # result m_mo method 1: 0.9048488091483489
    # result m_mo method 2: 0.8832390260631001

    best_mo = max(result_method_1, result_method_2)
    print('best overlap:', best_mo)

    # dementsprechend liefert method 1 die bessere segmentierung

    # teilaufgabe b thresholding

    i_and_mo = find_best_threshold(original, reference)
    best_threshold_for_max_mo = max(i_and_mo)
    print(f"the best mutual overlap metric with {best_threshold_for_max_mo[0]} is found when thresholding the image "
          f"with T = {best_threshold_for_max_mo[1]}")

    # overlap_108: 0.9754998462011689
