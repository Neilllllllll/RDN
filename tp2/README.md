# 📄 Fiche — Perceptron & Rétropropagation

## 1. Modèle du perceptron

### Somme
$$
s = \sum_{i=1}^{n} w_i x_i + b
$$

- $x_i$ : ième entrée  
- $w_i$ : poids associé  
- $b$ : biais  
- $s$ : somme

> Peu représenter une droite, un plan ou un hyperplan en fonction de $i$
---

### Fonction d’activation (sigmoïde)
$$
y = \phi(s) = \frac{1}{1 + e^{-s}}
$$

- $y$ : sortie du neurone
- $s$ : entrée de la sigmoïde  

> Dérivable en tout point 

---

## 2. Fonction de coût

### Erreur pour un exemple
$$
E = \frac{1}{2}(y_d - y)^2
$$

- $y_d$ : valeur attendue (du dataset)
- $y$ : sortie du modèle  

---

### Erreur moyenne sur un dataset
$$
E = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{2}(y_{d_i} - y_i)^2
$$

---

## 3. Dérivées intermédiaires

### Dérivée de l’erreur par rapport à la sortie
$$
\frac{\partial E}{\partial y} = y - y_d
$$

---

### Dérivée de la sigmoïde 
$$
\frac{\partial y}{\partial s} = y(1 - y)
$$

---

### 🔹 Dérivée de la somme par rapport au poids
$$
\frac{\partial s}{\partial w_i} = x_i
$$

---

## 4. Rétropropagation (gradient)

### 🔹 Règle de la chaîne
$$
\frac{\partial E}{\partial w_i} =
\frac{\partial E}{\partial y}
\cdot
\frac{\partial y}{\partial s}
\cdot
\frac{\partial s}{\partial w_i}
$$

---

### 🔹 Résultat final
$$
\frac{\partial E}{\partial w_i} = (y - y_d)\, y(1 - y)\, x_i
$$

---

## 6. Mise à jour des poids

$$
w_i \leftarrow w_i - \eta \frac{\partial E}{\partial w_i}
$$

- $\eta$ : pas d'apprentissage