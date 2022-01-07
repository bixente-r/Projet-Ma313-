"""
Fichier de test du projet Ma313

Ne lancer qu'une partie à la fois (mettre entre """ """ les autres parties) pour ne pas saturer la mémoire

Auteur : Maxime Gosselin
Classe : 3PF2
Groupe : Maxime Gosselin - Jimmy Hoarau - Matthieu Janiaut - Antonin Maitre
"""

import lib_projet as lib
import  numpy as np

A1 = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
A2 = np.array([[3, 0, 1], [0, 7, 0], [1, 0, 3]])
Msdp = lib.gen_matrice_sdp()
Hil3 = lib.Hilbert(3)
Hil10 = lib.Hilbert(10)
Ms = lib.gen_matrice_sym()
M = lib.gen_matrice()


###### Méthode de la puissane itérée ######
print("Matrice A1 sujet")
lib.affichage_méthode1(A1)
lib.affichage_norme1(A1)
print("####################################")
print("Matrice A2 sujet")
lib.affichage_méthode1(A2)
lib.affichage_norme1(A2)
print("####################################")
print("Matrice symétrique définie positive aléatoire")
lib.affichage_méthode1(Msdp[0])
lib.affichage_norme1(Msdp[0])
print("####################################")
print("Matrice symétrique définie positive aléatoire")
lib.affichage_méthode1(Msdp[1])
lib.affichage_norme1(Msdp[1])
print("####################################")
print("Matrice de Hilbert n=3")
lib.affichage_méthode1(Hil3)
lib.affichage_norme1(Hil3)
print("####################################")
print("Matrice de Hilbert n=10")
lib.affichage_méthode1(Hil10)
lib.affichage_norme1(Hil10)


###### Méthode de la puissane inverse ######

print("Matrice A1 sujet")
lib.affichage_méthode2(A1)
lib.affichage_cond2(A1)
print("####################################")
print("Matrice A2 sujet")
lib.affichage_méthode2(A2)
lib.affichage_cond2(A2)
print("####################################")
print("Matrice symétrique définie positive aléatoire")
lib.affichage_méthode2(Msdp[0])
lib.affichage_cond2(Msdp[0])
print("####################################")
print("Matrice symétrique définie positive aléatoire")
lib.affichage_méthode2(Msdp[1])
lib.affichage_cond2(Msdp[1])
print("####################################")
print("Matrice de Hilbert n=3")
lib.affichage_méthode2(Hil3)
lib.affichage_cond2(Hil3)
print("####################################")
print("Matrice de Hilbert n=10")
lib.affichage_méthode2(Hil10)
lib.affichage_cond2(Hil10)


###### Adaptation au cas d'une matrice quelconque ######

A3 = np.array([[1, 1], [1, -2]])
A4 = np.array([[0, 1], [1, 0]])

print("Matrice A3 sujet")
lib.affichage_méthode3(A3)
print("####################################")
print("Matrice A4 sujet")
lib.affichage_méthode3(A4)
print("####################################")
print("Matrice symétrique aléatoire")
lib.affichage_méthode3(Ms[0])
print("####################################")
print("Matrice symétrique aléatoire")
lib.affichage_méthode3(Ms[1])


###### Introduction à la méthode de diagonalisation par QR ######
print("Matrice A1 sujet (symétrique définie positive)")
lib.affichage_méthode4(A1)
print("####################################")
print("Matrice symétrique définie positive aléatoire")
lib.affichage_méthode4(Msdp[0])
print("####################################")
print("Matrice symétrique aléatoire")
lib.affichage_méthode4(Ms[0])
print("####################################")
print("Matrice symétrique aléatoire")
lib.affichage_méthode4(Ms[1])
print("####################################")
print("Matrice aléatoire")
lib.affichage_méthode4(M[0])
print("####################################")
print("Matrice aléatoire")
lib.affichage_méthode4(M[1])
