"""
Librairie des fonctions du projet Ma313

Auteur : Maxime Gosselin
Classe : 3PF2
Groupe : Maxime Gosselin - Jimmy Hoarau - Matthieu Janiaut - Antonin Maitre
"""
import numpy as np


def matrice_def_pos(A):
    """
    Vérifie qu'une matrice est définie positive
    Renvoie booléen
    """
    if np.all(np.linalg.eigvals(A) > 0):
        mdp = True
    else:
        print("la matrice n'est pas défninie positive")
        mdp = False
    return mdp


def matrice_sym(A):
    """
    Vérifie qu'une matrice est symétrique
    Renvoie booléen
    """
    At = np.transpose(A)
    # print(A)
     #print(At)
    if (np.allclose(A, A.T) == True):
        ms = True
    else:
        print("la matrice n'est pas symétrique")
        ms = False
    return ms




###### Méthode de la puissance itérée ######

def méthode_puissance_itérée(A, epsilon=10**-10, Nitermax=250):
    """
    Applique la méthode de la puissance itérée à une matrice

    Paramètres : Matrice carré - epsilon float : << 1 - Nombre max d'itération

    Sortie : 
        - Approximation du vecteur propre associé
        - Approximation de la valeur propre
        - Nombre d'itérations
        - Dernier écart calculé
    """
    mdp = matrice_def_pos(A)
    ms = matrice_sym(A)
    if (mdp != True or ms != True):
        pass
    else:
        n,n = np.shape(A)
        # x0 = np.zeros((n,1))
        # x0[:,0] = 1
        # x = x0
        x = np.random.rand(n, 1)
        norme_x = np.linalg.norm(x)
        w0 = (1/norme_x)*x
        Niter = 0
        #print(np.linalg.norm(x), "\n")
        # w0 = (1/np.linalg.norm(x))*x
        w = w0
        err = epsilon + 1

        while(Nitermax > Niter and err > epsilon):
            wk = w
            #print("wk = ", wk, "\n")
            c = np.dot(A,wk)
            #print("norm c = ", np.linalg.norm(c))
            #print("c = ", c,"\n")
            w = (1/np.linalg.norm(c))*c
            #print("wk+1 = ",w,"\n")
            Niter += 1
            err = np.linalg.norm(w-wk)
            #print(Niter,err,"---------------------","\n")
            norm_c = np.linalg.norm(c)
    return w, norm_c, Niter, err


def norme_euclidienne(A):
    """
    Calcule la norme euclidienne d'une matrice carré à partir de la méthode de la puissance itérée
    
    Paramètre : Matrice carré

    Sortie :
        - Norme calculée à partir de la puissance itérée
        - Norme euclidienne de numpy
    """
    At = np.transpose(A)
    a = np.dot(At,A)
    vp_max = méthode_puissance_itérée(a, 10**-10, 50)[1]
    norm_A_calc = np.sqrt(vp_max)
    norm_a_np = np.linalg.norm(A,2)
    return norm_A_calc, norm_a_np





###### Méthode de la puissance inverse ######

def méthode_puissance_inverse(A, epsilon=10**-10, Nitermax=250):
    """
    Applique la méthode de la puissance inverse à une matrice

    Paramètres : Matrice carré - epsilon : float << 1 - Nombre max d'itération

    Sortie : 
        - Approximation du vecteur propre associé
        - Approximation de la plus grande valeur propre de la matrice inverse
        - Approximation de la plus petite valeur propre de la matrice
        - Nombre d'itérations
        - Dernier écart calculé
    """
    mdp = matrice_def_pos(A)
    ms = matrice_sym(A)
    a = np.linalg.inv(A)
    if (mdp != True or ms != True):
        print("")
    else:
        n,n = np.shape(A)
        # x0 = np.zeros((n,1))
        # x0[:,0] = 1
        # x = x0
        x = np.random.rand(n, 1)
        norme_x = np.linalg.norm(x)
        w0 = (1/norme_x)*x
        Niter = 0
        #print(np.linalg.norm(x), "\n")
        # w0 = (1/np.linalg.norm(x))*x
        w = w0
        err = epsilon + 1

        while(Nitermax > Niter and err > epsilon):
            wk = w
            #print("wk = ", wk, "\n")
            c = np.dot(a,wk)
            #print("norm c = ", np.linalg.norm(c))
            #print("c = ", c,"\n")
            w = (1/np.linalg.norm(c))*c
            #print("wk+1 = ",w,"\n")
            Niter += 1
            err = np.linalg.norm(w-wk)
            #print(Niter,err,"---------------------","\n")
            norm_c = np.linalg.norm(c)
            vp = 1/norm_c
    return w, norm_c, vp, Niter, err


def norme_euclidienne_2(A):
    """
    Calcule la norme euclidienne de l'inverse d'une matrice carré à partir de la méthode de la puissance inverse
        
    Paramètre : Matrice carré

    Sortie :
        - Norme calculée à partir de la puissance inverse
        - Norme euclidienne de numpy
    """
    At = np.transpose(A)
    a = np.dot(At,A)
    vp_max = méthode_puissance_inverse(a, 10**-10, 50)[1]
    norm_A_calc = np.sqrt(vp_max)
    norm_a_np = np.linalg.norm(np.linalg.inv(A),2)

    return norm_A_calc, norm_a_np


def conditionnement(A):
    """
    Calcule le conditionnement d'une matrice carré à partir de la méthode de la puissance itérée et inverse
        
    Paramètre : Matrice carré

    Sortie :
        - Conditionnement calculé à partir des méthodes
        - Conditionnement de numpy
    """
    norm_A = norme_euclidienne(A)[0]
    norm_inv_A = norme_euclidienne_2(A)[0]
    cond = norm_inv_A * norm_A
    condd = np.linalg.cond(A)
    return cond, condd

###### Adaptation au cas d'une matrice quelconque ######

def méthode_puissance_itérée2(A, epsilon=10**-10, Nitermax=500):
    """
    Applique la méthode de la puissance itérée à une matrice quelconque

    Paramètres : Matrice carré - epsilon float : << 1 - Nombre max d'itération

    Sortie : 
        - Approximation du vecteur propre associé
        - Approximation de la valeur propre
        - Nombre d'itérations
        - Dernier écart calculé
    """
    # print("\n")
    n,n = np.shape(A)
    # x0 = np.zeros((n,1))
    # x0[:,0] = 1
    # x = x0
    x = np.random.rand(n, 1)
    norme_x = np.linalg.norm(x)
    w0 = (1/norme_x)*x
    Niter = 0
    #print(np.linalg.norm(x), "\n")
    # w0 = (1/np.linalg.norm(x))*x
    w = w0
    err = epsilon + 1

    while(Nitermax > Niter and err > epsilon):
        wk = w
        #print("wk = ", wk, "\n")
        c = np.dot(A,wk)
        #print("norm c = ", np.linalg.norm(c))
        #print("c = ", c,"\n")
        w = (1/np.linalg.norm(c))*c
        #print("wk+1 = ",w,"\n")
        Niter += 1
        err = np.linalg.norm(w-wk)
        #print(Niter,err,"---------------------","\n")
        # list_w.append(w)
        norm_c = np.linalg.norm(c)

    if Niter >= Nitermax:
        print("Le nombre maximal d'itération est dépassé.")
        wk = w
        c = np.dot(A, wk)*(-1)
        w = (1 / np.linalg.norm(c))*c
        err = np.linalg.norm(w-wk)
        norm_c = np.linalg.norm(c)

        if err < epsilon:
            cas = 1
        elif err >= epsilon:
            cas = 2
    elif err <= epsilon:
        cas=3
    return w, norm_c, Niter, err, cas

###### Introduction à la méthode de diagonalisation par QR ######

def diag_QR(A, epsilon=10**-10, Nitermax=250):
    """
    Applique la méthode de diagonalisation par QR

    Paramètres : Matrice carré - epsilon : float << 1 - Nombre max d'itération

    Sortie : 
        - Dernière matrice Ak calculée
        - Matrice de passage orthogonale P telle que Ak = Pt*A*P
        - Nombre d'itérations
        - Valeur maximale Mk des valeurs absolues des coefficients sous la diagonale de Ak
    """
    n,n = np.shape(A)
    Niter = 0
    Mk = epsilon + 1
    A0 = A
    a = A0
    list_Q = []
    while (Nitermax > Niter and Mk > epsilon):
        ak = a
        # print("ak", ak,"\n")
        Qk = np.linalg.qr(ak)[0]
        Rk = np.linalg.qr(ak)[1]
        list_Q.append(Qk)
        
        m = np.tril(ak)-np.diag(np.diag(ak))
        M = abs(m)
        Mk = M.max() 
        Niter += 1
        a = np.dot(Rk,Qk)
        # print("a", a, "\n")
    return ak, list_Q[0], Niter, Mk





###### Génération de matrices ######

def gen_matrice_sdp(min=2, max=10, pas=5):
    """
    Génère une liste de matrice symétrique définie positive
    """
    list_Msdp = []
    for i in range(min, max, pas):
        A = np.random.rand(i,i)
        As = (A + A.T)/2
        # print(As,"\n")
        At = np.transpose(As)
        # print(At,"\n")
        Asdp = np.dot(As, At)
        # print(Asdp)
        dp = matrice_def_pos(Asdp)
        if dp == True:
            list_Msdp.append(Asdp)
    return list_Msdp


def Hilbert(n):
    """
    Renvoie la matrice de HIlbert de taille n
    """
    H = np.zeros((n, n))
    for i in range(0,n):
        for j in range(0, n):
            H[i,j] = 1.0/(i+j+1)
    return H


def gen_matrice_sym(min=2, max=10, pas=5):
    """
    Génère une liste de matrice symétrique
    """
    list_Msym = []
    for i in range(min, max, pas):
        A = np.random.rand(i,i)
        As = (A + A.T)/2
        s = matrice_sym(As)
        if s == True:
            list_Msym.append(As)
    return list_Msym


def gen_matrice(min=2, max=10, pas=5):
    """
    Génère des matrices aléatoires
    """
    list_M = []
    for i in range(min, max, pas):
        A = np.random.rand(i,i)
        list_M.append(A)
    return list_M


def affichage_méthode1(A):
    mpi_A = méthode_puissance_itérée(A)
    eig_A = np.linalg.eig(A)
    print("\nMatrice : \n", A)
    print("\n- valeur propre max de A : ", mpi_A[1], "\n- vecteur propre associé : \n", mpi_A[0], "\n- nombre d'itération : ", mpi_A[2], "\n- erreur : ", mpi_A[3])
    print("\nDiagonalisation de A avec numpy : ")
    print("\n- valeur propre de A : \n", eig_A[0], "\n- matrice de passage associée : \n", eig_A[1])

def affichage_norme1(A):
    norme = norme_euclidienne(A)
    print("\n- norme calculée à partir de la puissance itérée :\n", norme[0])
    print("- norme calculée à de la bibliothèque numpy :\n", norme[1])


def affichage_méthode2(A):
    mpi_A = méthode_puissance_inverse(A)
    eig_A = np.linalg.eig(A)
    print("\nMatrice : \n", A)
    print("\n- valeur propre max de A**-1 : ", mpi_A[1],"\n- valeur propre min de A : ", mpi_A[2], "\n- vecteur propre associé : \n", mpi_A[0], "\n- nombre d'itération : ", mpi_A[3], "\n- erreur : ", mpi_A[4])
    print("\nDiagonalisation de A avec numpy : ")
    print("\n- valeurs propres de A : \n", eig_A[0], "\n- matrice de passage associée : \n", eig_A[1])

def affichage_cond2(A):
    cond = conditionnement(A)
    print("\n- conditionnement calculée à partir de la puissance inverse :\n", cond[0])
    print("- conditionnement calculée à de la bibliothèque numpy :\n", cond[1])


def affichage_méthode3(A):
    mpi_A = méthode_puissance_itérée2(A)
    eig_A = np.linalg.eig(A)
    cas = mpi_A[4]
    print("\nMatrice : \n", A)
    if cas == 1:
        print("\nLa suite wk diverge mais la suite (-1)*wk converge")
        print("\n- valeur propre max de A : ", mpi_A[1], "\n- vecteur propre associé : \n", mpi_A[0], "\n- nombre d'itération : ", mpi_A[2], "\n- erreur : ", mpi_A[3])
    elif cas == 2:
        print("\nLes suites wk et (-1)^k*wk divergent\n")
    elif cas == 3:
        print("\nLa suite wk converge")
        print("\n- valeur propre max de A : ", mpi_A[1], "\n- vecteur propre associé : \n", mpi_A[0], "\n- nombre d'itération : ", mpi_A[2], "\n- erreur : ", mpi_A[3])
    print("\nDiagonalisation de A avec numpy : ")
    print("\n- valeurs propres de A : \n", eig_A[0], "\n- matrice de passage associée : \n", eig_A[1])

def affichage_méthode4(A):
    mpi_A = diag_QR(A)
    eig_A = np.linalg.eig(A)
    print("\nMatrice : \n", A)
    print("\n- matrice triangulaire sup Ak : \n", mpi_A[0], "\n- matrice de passage orthogonale P : \n", mpi_A[1], "\n- nombre d'itération : ", mpi_A[2], "\n- valeur maximale Mk : ", mpi_A[3])
    print("\nDiagonalisation de A avec numpy : ")
    print("\n- valeurs propres de A : \n", eig_A[0], "\n- matrice de passage associée : \n", eig_A[1])

