import numpy as np
import matplotlib.pyplot as plt

def afficherPoints(D):
    colors = ["red" if o == 0 else "green" for o in D[:,3]]
    row_x, row_y, row_z = D[:,0], D[:,1], D[:,2]
    ax.scatter(row_x,row_y,row_z,c = colors)

def afficherPlan(W): 
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)            
    ax.set_zlim(-5, 5)
    colors = ["red" if o == 0 else "green" for o in D[:,3]]
    row_x, row_y, row_z = D[:,0], D[:,1], D[:,2]
    ax.scatter(row_x,row_y,row_z,c = colors)
    mesh_range = np.arange(-1.2,1.2,0.1)
    mesh_x,mesh_y = np.meshgrid(mesh_range,mesh_range)
    z = -1 / W[2] * (W[0]*mesh_x + W[1]*mesh_y + W[3]) 
    ax.plot_surface(mesh_x,mesh_y,z,alpha=0.4)
    plt.pause(0.5)

# def calculErreur(D, W): 
#     erreurs = 0
#     for i in range(len(D)):
#         X = np.append(D[i,:3], 1).reshape(1, -1) # X = (x,y,z,1)
#         y_d = D[i,-1]
#         y = neurone(X,W)
#         if y != y_d: # 1.3 - Cette ligne n'est plus correcte !
#             erreurs += 1
#     print("Erreur absolue :", erreurs)
#     return erreurs


def neurone(X, W):
    # Somme
    for i in len(W):
        s += W[i]*X[i]
    # Function sigmoide
    y = 1 / (1 + np.exp(s))
    return y


def entrainement(D, nb_iter=1000, eta=0.1, batch_size=10):

    W = np.random.rand(4)  
    N = D.shape[0]

    for t in range(nb_iter):
        indices = np.random.randint(0, N, size=batch_size)
        batch = D[indices]
        grad_total = np.zeros_like(W)

        for ligne in batch:
            X = np.append(ligne[:3], 1).reshape(1, -1) # X = (x,y,z,1)
            y_d = ligne[-1]
            y = neurone(X, W)

            grad_i = np.zeros_like(W)
            for i in len(grad_i):
                grad_i[i]= ((y - y_d) * y(1-y) * X[i])
                grad_total[i] += grad_i[i]

            grad_moyen = ?? # 1.2.c
            W = ?? # 1.2.d

        if t % 100 == 0:
            afficherPlan(W)
            calcul(D,W)

    return W

# def utiliserNeurone(X, W):
#     X_aug = np.append(X, 1)
#     y = neurone(X_aug, W)
#     # ?? # 1.4 - Affiche la couleur prédite du point


# --- Programme principal ---
D = np.loadtxt("donnees.csv", delimiter=",")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

afficherPoints(D)

W = entrainement(D, nb_iter=1000, eta=0.1, batch_size=10)

