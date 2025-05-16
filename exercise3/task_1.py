import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from skimage import io




# region identification mit floodfill
# eingabe bereits binäres bild, durch thresholding erzeugt
# vordergrundpixel = 255, hintergrundpixel = 0

BASE_DIRECTORY = '/Users/laurinerichlitzki/laurinesrepository/semester6/img'

# anwendung floodfill
# einfärben in unterschiedlichen farbe je nach object


def RegionLabeling(image):
    height, width = image.shape[:2]
    labeled = np.zeros_like(image, dtype=int)
    label = 2  # Start bei 2, 0 = Hintergrund, 1 = noch nicht gelabelt
    for i in range(height):
        for j in range(width):
            if image[i, j] == 1 and labeled[i, j] == 0: # 1 noch nicht zugeordnet/gelabelt, 0 im endbild
                FloodFill(image, labeled, i, j, label)
                label += 1
    return labeled


def FloodFill(image, labeled, i, j, label):
    height, width = image.shape[:2]
    if not (0 <= i < height and 0 <= j < width):
        return # bildränder!!
    if image[i, j] == 1 and labeled[i, j] == 0:
        labeled[i, j] = label
        # 4nb, 8er macht nicht unbedingt einen unterschied
        FloodFill(image, labeled, i-1, j, label) # oben
        FloodFill(image, labeled, i+1, j, label) # unten
        FloodFill(image, labeled, i, j-1, label) # links
        FloodFill(image, labeled, i, j+1, label) # rechts


def label2rgb(label_img):
    h, w = label_img.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    num_labels = label_img.max()
    np.random.seed(0)

    for label in range(3, num_labels + 1):  # Label startet bei 2
        color = np.random.randint(0, 255, size=3)
        out[label_img == label] = color

    return out


if __name__ == '__main__':
    image = io.imread(os.path.join(BASE_DIRECTORY, 'embryos_binary.png'), as_gray=True)
    image = image.astype(int)
    plt.imshow(image, cmap='gray')
    plt.show()

    sys.setrecursionlimit(4000)

    labeled = RegionLabeling(image)

    labels = np.unique(labeled)
    print("vergebene labels:", labels)
    num_regions = len(labels) - (1 if 0 in labels else 0)
    print("anzahl regionen:", num_regions)

    gelabelte_pixel = np.sum(labeled > 0)
    print("anzahl gelabelter pixel:", gelabelte_pixel)

    for label in labels:
        count = np.sum(labeled == label)
        print(f"label {label}: {count} pixel")

    colored = label2rgb(labeled)
    plt.imshow(colored)
    plt.axis('off')
    plt.title('Region Labels')
    plt.show()
