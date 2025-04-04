import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

BASE_DIRECTORY = '/Users/laurinerichlitzki/laurinesrepository/semester6/img'

image = io.imread(os.path.join(BASE_DIRECTORY, 'cells.png'))


# teilaufgabe a bildränder, randstreifen von 10 intensität auf 0 -> rand
def rand(image, border):
    # Ränder setzen
    image[:border, :] = 0  # oben
    image[-border:, :] = 0  # unten
    image[:, :border] = 0  # links
    image[:, -border:] = 0  # rechts

    return image


# rand erstellen
border_size = 10
image_with_border = rand(image.copy(), border_size)


# teilaufgabe b bildhelligkewit um 19% erhöhen

def bildaufhellung(image, percent):
    image = image * percent
    image = np.clip(image, 0, 255) # randwerte clippen
    image = image.astype('uint8') # wieder von float zurück konvertieren
    return image


# helligkeit erhöhen
percent = 1.19
image_lighter = bildaufhellung(image_with_border, percent)

# bild anzeigen
plt.imshow(image_with_border, cmap=plt.cm.gray)
plt.show()

plt.imshow(image_lighter, cmap=plt.cm.gray)
plt.show()