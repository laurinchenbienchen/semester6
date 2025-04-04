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


#teilaufgabe b mittelwert der bildintensoitäten

def mittelwert(image):
    mittel = np.mean(image)
    print('Mittelwert:', mittel)


mittelwert(image)
