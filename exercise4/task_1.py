import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.color import label2rgb as sk_label2rgb
import cv2
from skimage import io
import os
import sys

BASE_DIRECTORY = '/Users/laurinerichlitzki/laurinesrepository/semester6/img'


# teilaufgabe a, geeignete segmentierung des bildes
# threshold otsu
def segmentation(img):
    T_otsu = skimage.filters.threshold_otsu(img)
    binary = np.where(img <= T_otsu, 0, 255).astype(np.uint8)
    return binary


# durch segmentierung skalierung aus dem bild entfernt

# teilaufgabe b detektion einzelobjekte, label-methode
def RegionLabeling(image):
    height, width = image.shape[:2]
    labeled = np.zeros_like(image, dtype=int)
    label = 2  # Start bei 2, 0 = Hintergrund, 1 = noch nicht gelabelt
    for i in range(height):
        for j in range(width):
            if image[i, j] == 1 and labeled[i, j] == 0:  # 1 noch nicht zugeordnet/gelabelt, 0 im endbild
                FloodFill(image, labeled, i, j, label)
                label += 1
    return labeled


def FloodFill(image, labeled, i, j, label):
    height, width = image.shape[:2]
    if not (0 <= i < height and 0 <= j < width):
        return  # bildränder!!
    if image[i, j] == 1 and labeled[i, j] == 0:
        labeled[i, j] = label
        # 4nb, 8er macht nicht unbedingt einen unterschied
        FloodFill(image, labeled, i - 1, j, label)  # oben
        FloodFill(image, labeled, i + 1, j, label)  # unten
        FloodFill(image, labeled, i, j - 1, label)  # links
        FloodFill(image, labeled, i, j + 1, label)  # rechts


def label2rgb(label_img):
    h, w = label_img.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    num_labels = label_img.max()
    np.random.seed(0)

    for label in range(3, num_labels + 1):  # Label startet bei 2
        color = np.random.randint(0, 255, size=3)
        out[label_img == label] = color

    return out


# teilaufgabe c/d relevante merkmale für einzelobjekte, fläche und ein weiteres relevantes merkmal zu form/rundheit
# und rausfiltern
def berechne_merkmale_und_filtere(labeled_img, min_area=150, min_circularity=0.7):
    props = skimage.measure.regionprops(labeled_img)
    gefilterte_labels = []

    print("Gefundene Objekte und Merkmale:")

    for region in props:
        area = region.area
        perimeter = region.perimeter
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        label = region.label
        print(f"objekt {label}: fläche = {area}, umfang = {perimeter:.2f}, rundheit = {circularity:.3f}")

        if area >= min_area and circularity >= min_circularity:
            gefilterte_labels.append(label)

    return gefilterte_labels

# teilaufgabe e visualisierung von den verbliebenen objekten mit ursprünglichen labels
def visualisiere_gefilterte(labeled_img, original_img, gefilterte_labels):
    overlay = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for label in gefilterte_labels:
        coords = np.column_stack(np.where(labeled_img == label))
        y, x = coords.mean(axis=0).astype(int)  # Schwerpunkt
        cv2.putText(overlay, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (154,205,50), 1, cv2.LINE_AA)

    plt.imshow(overlay)
    plt.title('gefilterte objekte mit label-overlay')
    plt.axis('off')
    plt.show()

def zeige_gefilterte_labels_farbig(labeled_img, gefilterte_labels):
    maske = np.isin(labeled_img, gefilterte_labels) * labeled_img  # nur gefilterte Labels
    farbig = sk_label2rgb(maske, bg_label=0)  # echte skimage-Funktion

    plt.imshow(farbig)
    plt.title("gefilterte (runde) objekte farbig")

    for label in gefilterte_labels:
        coords = np.column_stack(np.where(labeled_img == label))
        y, x = coords.mean(axis=0).astype(int)
        plt.text(x, y, str(label), color='black', fontsize=8, ha='center', va='center',
                 bbox=dict(facecolor='black', edgecolor='none', pad=1.0, alpha=0.5))

    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    img = io.imread(os.path.join(BASE_DIRECTORY, 'embryos_binary.png'), as_gray=True)

    seg_img = segmentation(img)
    plt.imshow(seg_img, cmap='gray')
    plt.title('Segmentiertes Bild')
    plt.axis('off')
    plt.show()

    sys.setrecursionlimit(4000)

    labeled = RegionLabeling(seg_img // 255)

    colored = label2rgb(labeled)
    plt.imshow(colored)
    plt.axis('off')
    plt.title('Region Labels')
    plt.show()

    labels = np.unique(labeled)
    print("vergebene labels:", labels)
    num_regions = len(labels) - (1 if 0 in labels else 0)
    print("anzahl regionen:", num_regions)

    gelabelte_pixel = np.sum(labeled > 0)
    print("anzahl gelabelter pixel:", gelabelte_pixel)

    for label in labels:
        count = np.sum(labeled == label)
        print(f"label {label}: {count} pixel")

    gefilterte = berechne_merkmale_und_filtere(labeled)

    visualisiere_gefilterte(labeled, img, gefilterte)

    gefilterte_labels = berechne_merkmale_und_filtere(labeled, min_area=100, min_circularity=0.7)
    zeige_gefilterte_labels_farbig(labeled, gefilterte_labels)