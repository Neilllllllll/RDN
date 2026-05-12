import numpy as np
import matplotlib.pyplot as plt
from utils import get_images_and_labels, visualisation_poids, charger_et_vectoriser_image, plot_precision

def afficher_exemples(images,labels):
    plt.figure(1)
    for i in range(36):
        ind = np.random.randint(images.shape[0])
        image = images[ind]
        plt.subplot(6, 6, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(str(labels[ind]), fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def afficher_indice(images, labels, ind):
    plt.figure(1)
    image = images[ind]
    plt.imshow(image, cmap='gray')
    plt.title(str(labels[ind]), fontsize=8)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# # Calcule la sortie désirée Y_d
def calcul_sd(label):
    Y_d = np.zeros((10, 1))
    Y_d[label] = 1
    return Y_d

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(E, W1, W2):
    E_biais = np.vstack((E, [1]))

    s1 = W1 @ E_biais
    r1 = sigmoid(s1)

    r1_biais = np.vstack((r1, [1]))

    s2 = W2 @ r1_biais
    r2 = sigmoid(s2)

    Y = r2
    Y_c = r1

    return Y, Y_c

# # Rétropropagation : fonction complète, ne pas modifier !
def backprop(Y, Y_d, Y_c, eta, W1, W2, E):

    # ---- Couche 2 ----
    # Ajout biais à S1 pour correspondre à la taille de W2
    S1_biais = np.vstack((Y_c,[1]))                        

    # Calcul de sigma pour la couche de sortie
    sigma2 = (Y - Y_d) * Y * (1 - Y)               

    # Mise à jour des poids de la couche 2
    W2 -= eta * sigma2 @ S1_biais.T                         

    # ---- Couche 1 ----
    # Propagation de l'erreur pour chaque neurone de la couche 1
    sigma1 = (W2[:, 1:].T @ sigma2) * Y_c * (1 - Y_c)        

    # Ajout biais à E
    E_biais = np.vstack((E,[1]))

    # Mise à jour des poids de la couche 1
    W1 -= eta * sigma1 @ E_biais.T                          

    return W1, W2

# # Fonction qui calcule la précision du réseau
def calcul_precision(W1,W2):
    images, labels = get_images_and_labels("t10k") # !
    N = images.shape[0]
    n_corrects = 0
    for ind in range(N):
        
        E = images[ind].reshape(-1, 1)
        Y, _ = forward(E, W1, W2)

        prediction = np.argmax(Y)

        if prediction == labels[ind]:
            n_corrects += 1
        
    return n_corrects / images.shape[0]


# # Fonction qui teste le réseau sur un échantillon d'images inconnues    
# def test_reseau(W1, W2):
    
#     images, labels = get_images_and_labels("t10k") # !
    
#     plt.figure(6)
#     plt.clf()
#     plt.colormaps()
#     plt.gray()

#     for i in range(36):
#         plt.subplot(6, 6, i+1)
#         ind = np.random.randint(images.shape[0])
#         E = images[ind].reshape(-1, 1)
        
#         ??
#         prediction = ?? # Quel est le chiffre prédit par le réseau ?
#         confiance = ?? # Quel est la confiance du réseau su le chiffre prédit ?

#         image = E.reshape(28, 28)
#         plt.imshow(image, cmap='gray')
#         plt.title(f"{prediction} ({confiance:.2f})", fontsize=8)
#         plt.axis('off')

#     plt.tight_layout()
#     plt.show()
#     plt.pause(0.1)

# # %%

def entrainer(images,labels,nb_iter=10000,eta=0.1):
    
    # Initialisation des paramètres 
    # 2.1 - Nombres de neurones par couche
    n_0 = 784 # ! Couche d'entrée
    n_1 = 32 # ! Couche cachée
    n_2 = 10 # ! Couche de sortie
    
    # 2.2 - Matrices de paramètres
    W1 = np.random.randn(n_1, n_0 + 1) * np.sqrt(1 / n_0)
    W2 = np.random.randn(n_2, n_1 + 1) * np.sqrt(1 / n_1)
    
    # Variables pour le diagnostic
    historique_precision = []
    
    # Entraînement
    for iter in range(nb_iter):
    
        # Sélection aléatoire d’un échantillon
        ind = np.random.randint(images.shape[0])
        
        # Construction de l'entrée du réseau
        E = images[ind].reshape(-1, 1)

        # Calcul de la sortie désirée
        Y_d = calcul_sd(labels[ind])
    
        # Entrainement du réseau (1 itération)
        Y, Y_c = forward(E, W1, W2)

        W1, W2 = backprop(Y, Y_d, Y_c, eta, W1, W2, E)
        
        # Affichage de l'évolution de la précision du réseau
        if (iter + 1) % 1000 == 0:
            prec = calcul_precision(W1,W2)
            historique_precision.append(prec)
            plot_precision(historique_precision, intervalle=1000)
            print(f"Precision : {calcul_precision(W1,W2)}")
       
        # Visualisation du modèle interne du réseau
        if (iter + 1) % 1000 == 0:
            visualisation_poids(W1,W2)
            plt.pause(0.05)

    return W1,W2

# def test_paint(image, W1, W2):
#     # Chargement et conversion de l’image
#     E = charger_et_vectoriser_image(image)
    
#     # Affichage de l’image
#     plt.figure(7)
#     plt.imshow(E.reshape(28, 28), cmap='gray')
#     plt.axis('off')

#     # Prédiction par le réseau
#     ??
#     prediction = ??
#     confiance = ??

#     plt.title(f"Il s'agit d'un {prediction} avec une probabilité de {confiance:.2f}")
#     plt.show()
    

# ####### ---------------------------- Point d'entrée ------------------------------ #######
# Chargement de la base de test
images, labels = get_images_and_labels("train") # 

entrainer(images=images, labels=labels)

# Affichage de 36 échantillons aléatoires du dictionnaire
# afficher_exemples(images,labels)



