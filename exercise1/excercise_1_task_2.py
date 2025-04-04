import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

BASE_DIRECTORY = '/Users/laurinerichlitzki/laurinesrepository/semester6/img'

image = io.imread(os.path.join(BASE_DIRECTORY, 'cells.png'))


# teilaufgabe a bildbreite und höhe ausgeben

def bilddaten(image):
    height = image.shape[0]
    width = image.shape[1]
    print('Bildhöhe:', height, 'Bildbreite:', width)


bilddaten(image)
# Bildhöhe:489 Bildbreite:640


#teilaufgabe b mittelwert der bildintensoitäten

def mittelwert(image):
    mittel = np.mean(image)
    print('Mittelwert:', mittel)


mittelwert(image)
# Mittelwert: 196.69835


# teilaufgabe c intensitätshistogramm
def histo_intesity(image):
    # histo berechnen
    histo, bins = np.histogram(image, bins=256, range=(0, 256))
    # plotten des histo
    plt.figure(figsize=(8, 5))
    plt.bar(bins[:-1], histo, color='green', width=1)
    plt.plot(histo, color='green')
    plt.title("Intensitätshistogramm")
    plt.xlabel("Pixelwert")
    plt.ylabel("Anzahl der Pixel")
    plt.show()

histo_intesity(image)

# teilaufgabe d
# sinvoller schwellwert : 190 (ca. zwischen 180 und 200)
# Histogrammtyp: bimodales histogramm
