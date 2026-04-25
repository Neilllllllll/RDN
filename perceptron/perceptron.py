import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, w = np.random.rand(3), b = np.random.rand(1), alpha = 0.1):
        self.w = w
        self.b = b
        self.alpha = alpha
        self.ax = plt.figure().add_subplot(111, projection='3d')
    
    def process_single_point(self, x1, x2, x3):
        s = x1*self.w[0] + x2*self.w[1] + x3*self.w[2] + self.b
        sortie = 0
        if s > 0:
            sortie = 1
        return sortie
    
    def train(self, D, nb_iter): # 2.1 - Identifier l'entrée principale de l'entrainement
        for i in range(nb_iter): 
            # 2.3.a - Charger aléatoirement une ligne du dictionnaire
            valeur_aleatoire = D[np.random.randint(len(D))]
            # 2.3.b - Construire le vecteur d'entrées X et la sortie désirée y_d à partir de la ligne choisie
            x1 = valeur_aleatoire[0]
            x2 = valeur_aleatoire[1]
            x3 = valeur_aleatoire[2]
            # 2.4 - Calculer la sortie prédite y
            y_p = self.process_single_point(x1, x2, x3)
            # 2.5.b - Mettre à jour les poids
            self.w[0] = self.w[0] + self.alpha*(valeur_aleatoire[3] - y_p) * valeur_aleatoire[0]
            self.w[1] = self.w[1] + self.alpha*(valeur_aleatoire[3] - y_p) * valeur_aleatoire[1]
            self.w[2] = self.w[2] + self.alpha*(valeur_aleatoire[3] - y_p) * valeur_aleatoire[2]
            # 2.5.c - Mettre à jour le biais
            self.b = self.b + self.alpha*(valeur_aleatoire[3] - y_p)
            
            if(i%100 == 0): # 2.7.d - A quoi correspond ce 100 ? (n'hésitez pas à le changer)
                self.afficherPlan(self.w ,self.b)
                error = self.calculerErreur(D)
                if error == 0: 
                    print("Plus d'erreur entrainement terminé.")
                    return self.w, self.b
                if error < 10 and self.alpha != 0.001:
                    print("Réduction de alpha", 0.001)
                    self.alpha = 0.001
                elif error < 4 and self.alpha != 0.0001:
                    print("Réduction de alpha : ", 0.0001)

    # Parcous tout le tableau est compte le nombre de réponse fausse
    def calculerErreur(self, D):
        erreurs = 0
        for i in range(len(D)):
            x, y, z, y_d = D[i]
            y = self.process_single_point(x,y,z)
            if y != y_d:
                erreurs += 1
        print("Erreur absolue :", erreurs)
        return erreurs
    
    def afficherPlan(self, W,b):
        self.ax.clear()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)            
        self.ax.set_zlim(-5, 5)
        colors = ["red" if o == 0 else "green" for o in D[:,3]]
        row_x, row_y, row_z = D[:,0], D[:,1], D[:,2]
        self.ax.scatter(row_x,row_y,row_z,c = colors)
        mesh_range = np.arange(-1.2,1.2,0.1)
        mesh_x,mesh_y = np.meshgrid(mesh_range,mesh_range)
        z = -1 / W[2] * (W[0]*mesh_x + W[1]*mesh_y + b) # 2.7.b - Que fait cette ligne ?
        self.ax.plot_surface(mesh_x,mesh_y,z,alpha=0.4)
        plt.pause(0.5)

    def afficherPoints(self, D):
        colors = ["red" if o == 0 else "green" for o in D[:,3]]
        row_x, row_y, row_z = D[:,0], D[:,1], D[:,2]
        self.ax.scatter(row_x,row_y,row_z,c = colors)
    
    def utiliserNeurone(self, x1, x2, x3):
        if(self.process_single_point(x1, x2, x3) == 1):
            print("Le point x : ", x1 , " y : ",  x2 , " z : ", x3, " est rouge")
            return
        print("Le point x : ", x1 , " y : ",  x2 , " z : ", x3, " est vert")

mod = Perceptron()
D = np.loadtxt("donnees.csv",delimiter=",")
weight, biais = mod.train(D = D, nb_iter = 100000)
print(weight, biais)
# Pour le jeux de donnée 2 résultat trouvé : weight [ 0.67192672  0.68456866 -0.00353624] biais [-0.33694855]
# Pour le jeu de donnée 1 résultat trouvé : weight [ 0.45314944 -0.24473053 -0.39786813] biais [-0.04536178]