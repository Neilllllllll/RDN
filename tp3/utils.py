import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import os
from PIL import Image
    
# Chargement d'un dataset (fonction complète, ne pas modifier)
def get_images_and_labels(dataset_name: str) -> (np.ndarray, np.ndarray):
    image_file = os.path.abspath(os.path.join(os.path.dirname(
        __file__), '.', 'datasets/' + dataset_name + '-images.idx3-ubyte'))
    image_array = idx2numpy.convert_from_file(image_file)/255

    label_file = os.path.abspath(os.path.join(os.path.dirname(
        __file__), '.', 'datasets/' + dataset_name + '-labels.idx1-ubyte'))
    label_array = idx2numpy.convert_from_file(label_file)

    return image_array, label_array

# Visualisation des poids du réseau (fonction complète, ne pas modifier)
def visualisation_poids(W1,W2):
    # Visualisation de W1 (neurones de la couche cachée)
    plt.figure(2)
    for i in range(W1.shape[0]):
        plt.subplot(6, 6, i+1)
        img = (W1[i, 1:].reshape(28, 28) + 1) / 2
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.pause(0.01)

    # Visualisation des caractéristiques apprises
    plt.figure(3)
    for i in range(W2.shape[0]-1):
        plt.subplot(3, 4, i+1)
        neurone = W2[i, 1:] @ W1[:, 1:]  
        img = (neurone.reshape(28, 28) + 1) / 2
        plt.imshow(img, cmap='gray')
        plt.title(str(i))
        plt.axis('off')
    plt.pause(0.01)

# Affiche la courbe de l'évolution de la précision pendant l'entraînement.
# (fonction complète, ne pas modifier)
def plot_precision(historique, intervalle):
    """
    Affiche la courbe de l'évolution de la précision pendant l'entraînement.

    history  : liste des précisions enregistrées (valeurs entre 0 et 1)
    interval : nombre d'itérations entre deux mesures de précision
    """
    iterations = [i * intervalle for i in range(1, len(historique)+1)]
    
    plt.figure(4)
    plt.plot(iterations, [p * 100 for p in historique])
    plt.xlabel("Itérations")
    plt.ylabel("Précision (%)")
    plt.title("Évolution de la précision pendant l'entraînement")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Convertit un fichier png en vecteur d'entrée du réseau 
# (fonction complète, ne pas modifier)
def charger_et_vectoriser_image(fichier):
    img = Image.open(fichier).convert('L')       # Niveau de gris
    img = np.array(img, dtype=np.float32) / 255.0   # Normalisation [0,1]
    return img.reshape(-1, 1)                        # (784, 1)
