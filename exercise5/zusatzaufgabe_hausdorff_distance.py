import numpy as np
from skimage import io
import os

BASE_DIRECTORY = '/Users/laurinerichlitzki/laurinesrepository/semester6/img'


def calculate_boundary_pixels(image):
    image = image.copy()
    image_boundary = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i, j] == 255:
                if image[i, j - 1] == 0 or image[i, j + 1] == 0 or image[i - 1, j] == 0 or image[i + 1, j] == 0:
                    image_boundary[i, j] = 1
    return image_boundary


def get_pixel_coordinates(mask):
    koordinaten = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                koordinaten.append((i, j))
    return koordinaten  # koordinaten als tupel


def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def directed_hausdorff(setA, setB):
    max_min_dist = 0
    for pointA in setA:
        min_dist = float('inf')
        for pointB in setB:
            dist = euclidean_distance(pointA, pointB)
            if dist < min_dist:
                min_dist = dist
        if min_dist > max_min_dist:
            max_min_dist = min_dist  #merken der größten der minimalen distanzen
    return max_min_dist


def hausdorff_distance(mask1, mask2):
    # berechenen randpixel
    boundary1 = calculate_boundary_pixels(mask1)
    boundary2 = calculate_boundary_pixels(mask2)

    # koordinaten der randpixel
    coords1 = get_pixel_coordinates(boundary1)
    coords2 = get_pixel_coordinates(boundary2)

    # gerichtete hausdorff distanz
    forward = directed_hausdorff(coords1, coords2)
    backward = directed_hausdorff(coords2, coords1)

    # symmetrische hd-distanz
    symmetric = max(forward, backward)

    return forward, backward, symmetric


if __name__ == '__main__':
    method_1 = io.imread(os.path.join(BASE_DIRECTORY, 'hd_method1.png'), as_gray=True)
    method_2 = io.imread(os.path.join(BASE_DIRECTORY, 'hd_method2.png'), as_gray=True)

    fwd, bwd, sym = hausdorff_distance(method_1, method_2)

    print("Hausdorff-Distanz für reale Segmentierungen:")
    print(f"  Vorwärts (Method1 → Method2): {fwd:.2f}")
    print(f"  Rückwärts (Method2 → Method1): {bwd:.2f}")
    print(f"  Symmetrisch: {sym:.2f}")
