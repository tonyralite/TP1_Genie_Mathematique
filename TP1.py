#-------------------------------------------------------------------------------
# Name:       TP 1 Génie mathématique Ma223
#
# Etudiant 1 : Kylian Forsans
# Etudiant 2 : Tony Raliterason
#
# Classe : 2PG
# Groupe : G1
#
#-------------------------------------------------------------------------------

import numpy as np
import time as time
import matplotlib.pyplot as plt

#*************************************************************
#                ****** PARTIE 1 ******
#*************************************************************

#############################################################
                    # Question 1 #
#############################################################

def ReductionGauss(Aaug):
   
    if len(Aaug) == len(Aaug[0])-1:
        n = len(Aaug)
       
        for i in range(0, n-1):
       
            for k in range(i+1, n):
                pivot = Aaug[k, i]/Aaug[i, i]
               
                for j in range(0,n+1):
                    Aaug[k, j] = Aaug[k, j] - pivot*Aaug[i, j]
                   
        return Aaug
   
    else:
        print("La matrice entrée n'est pas augmentée")

#############################################################
                    # Question 2 #
#############################################################

def ResolutionSystTriSup(Taug):
   
    n, m = np.shape(Taug)
   
    if m != n+1:
       
        print("La matrice entrée n'est pas augmentée")
   
    x = np.zeros(n)
   
    for i in range(n-1, -1, -1):
        somme = 0
       
        for k in range(i+1, n):
            somme = somme + x[k]*Taug[i, k]
       
        x[i] = (Taug[i, -1]-somme)/Taug[i, i]
   
    return x

#############################################################
                    # Question 3 #
#############################################################

def Gauss(A,B) :
   
    Aaug = np.column_stack([A, B])
    Taug = ReductionGauss(Aaug)
    x = ResolutionSystTriSup(Taug)
   
    return x

#Question 4 (à la fin du code)


#*************************************************************
#                ****** PARTIE 2 ******
#*************************************************************

#############################################################
                    # Question 1 #
#############################################################

def DecompositionLU(A):
    n, m = A.shape
    L = np.eye(n)
    U = np.copy(A)
   
    for i in range(0, n-1):
   
        for k in range(i+1, n):
           
            pivot = U[k, i] / U[i, i]
            L[k, i] = pivot
           
            for j in range(i, n):
       
                U[k, j] = U[k, j] - pivot*U[i, j]
   
    return(L,U)

#############################################################
                    # Question 2 #
#############################################################

def ResolutionLU(L, U, B):
    Y = Gauss(L, B)
    X = Gauss(U, Y)
    return X

def ResolutionSystTriInf(Taug):
    n, m = Taug.shape

    if m != n+1:
        print("La matrice entrée n'est pas augmentée")
    x = np.zeros(n)
    for i in range(n):
        somme = 0

        for k in range(i):
            somme = somme + x[k]*Taug[i, k]
        x[i] = (Taug[i, -1]-somme)/Taug[i, i]
   
    return x

#*************************************************************
#                ****** PARTIE 3 ******
#*************************************************************

#############################################################
                    # Question 1 #
#############################################################

def GaussChoixPivotPartiel(A, B):
   
    Aaug = np.column_stack([A, B])
    n, m = np.shape(Aaug)
    q = 0
   
    if m!= n+1:
        print("La matrice entrée n'est pas augmentée")
    else:
        for i in range(0, n-1):
            for k in range(i+1, n):
                for j in range(k, n):
                    if abs(Aaug[j, i]) > abs(Aaug[i, i]):
                        Q = np.copy(Aaug[i, :])
                        Aaug[i, :] = Aaug[j, :]
                        Aaug[j, :] = Q
                if Aaug[i, i] == 0:
                    print("Le pivot est nul")
                q = Aaug[k, i]/Aaug[i, i]
                Aaug[k, :] = Aaug[k, :] - q*Aaug[i, :]
    X = ResolutionSystTriSup(Aaug)

    return X

#############################################################
                    # Question 2 #
#############################################################

def GaussChoixPivotTotal(A, B):
   
    Aaug = np.column_stack([A, B])
    n, m = np.shape(Aaug)
    q = 0
   
    if m!= n+1:
        print("La matrice entrée n'est pas augmentée")
   
    else:
        for i in range(m-1):
            for k in range(i+1, m-1):
                for j in range(k, m-1):
                    if abs(Aaug[j, i]) > abs(Aaug[i, i]):
                        Q = np.copy(Aaug[i, :])
                        Aaug[i, :] = Aaug[j, :]
                        Aaug[j, :] = Q
                if Aaug[i, i] == 0:
                    print("Le pivot est nul")
                q = Aaug[k, i]/Aaug[i, i]
                Aaug[k, :] = Aaug[k, :] - q*Aaug[i, :]
    X = ResolutionSystTriSup(Aaug)

    return X

#*************************************************************
#                ****** APPLICATION ******
#*************************************************************
dim = 4
lim_inf = 10
lim_sup = 30

intervalle = lim_sup - lim_inf
A = np.random.randint(low=lim_inf, high=lim_sup, size=(dim, dim))
B = np.random.randint(low=lim_inf, high=lim_sup, size=(dim, 1))

# Affichage de la matrice A
print('A = \n', A)

# Affichage de la matrice B
print('B = ', B.ravel(), end="\n\n")

#****************************************************

print("*** Algorithme de Gauss ***")

# Affichage de la matrice augmentée
print("Aaug = \n", np.concatenate((A, B), axis=1), end="\n\n")

# Affichage de la triangulaire supérieure
print("Taug =\n", ReductionGauss(np.concatenate((A, B), axis=1)), end="\n\n")

# Affichage de la résolution du système
print("X =", Gauss(A, B), end='\n\n')

#****************************************************

print("*** Résolution LU ***")

(L, U) = DecompositionLU(A)

# Affichage de la matrice L
print("L =\n", L, end="\n\n")

# Affichage de la matrice U
print("U =\n", U, end="\n\n")

# Affichage de la matrice Y
print("Y =\n", ResolutionSystTriInf(ReductionGauss(np.concatenate((A, B), axis=1))), end='\n\n')

# Affichage de la résolution du système
print("X =", ResolutionLU(L, U, B), end='\n\n')

#****************************************************

print("*** Résolution Pivot Partiel ***")

#Affichage de la résolution du système
print("X =", GaussChoixPivotPartiel(A, B), end='\n\n')

#****************************************************

print("*** Résolution Pivot Total ***")

#Affichage de la résolution du système
print("X =", GaussChoixPivotTotal(A, B), end='\n\n')

#****************************************************

print("*** Résolution linalg.solve ***")

#Affichage de la résolution du système
print("X =", np.linalg.solve(A, B).ravel(), end='\n\n')

#*************************************************************
#                ****** GRAPHIQUES ******
#*************************************************************

def Courbe_de_temps():
    indices = []
    y = []
    y2 = []
    y3 = []
    y4 = []
   
    for i in range(100,500,200):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)
       

        temps_initial = time.time()
        Gauss(A, B)
        temps_final = time.time()
        temps = temps_final - temps_initial

        temps_initial2 = time.time()
        [L, U] = DecompositionLU(A)
        ResolutionLU(L, U, B)
        temps_final2 = time.time()
        temps2 = temps_final2 - temps_initial2


        temps_initial3 = time.time() 
        GaussChoixPivotPartiel(A, B)
        temps_final3 = time.time()
        temps3 = temps_final3 - temps_initial3

 
        temps_initial4 = time.time()
        GaussChoixPivotTotal(A, B)
        temps_final4 = time.time()
        temps4 = temps_final4 - temps_initial4

 
        indices.append(i)
        y.append(temps)
        y2.append(temps2)
        y3.append(temps3)
        y4.append(temps4)
   
    plt.plot(indices, y, color='r', label="Gauss")
    plt.plot(indices, y2, color='b', label="LU")
    plt.plot(indices, y3, color='g', label="Pivot Partiel")
    plt.plot(indices, y4, color='k', label="Pivot Total")

    plt.title("Temps de calcul de la matrice en fonction de sa dimension")

    plt.xlabel("Dimension")
    plt.ylabel("Temps (sec)")

    plt.legend()

    plt.show()

Courbe_de_temps()

def Courbe_d_erreur():
    indices = []
    y = []
    y2 = []
    y3 = []
    y4 = []

    for i in range(100, 500, 200):
        A = np.array(np.random.random(size=(i, i)))
        B = np.array(np.random.random(size=(i, 1)))
        C = np.copy(A)
       
 
        Gauss(A, B)
        erreur = np.linalg.norm(A@Gauss(A, B) - np.ravel(B))
 

        L, U = DecompositionLU(A)
        ResolutionLU(L, U, B)
        erreur2 = np.linalg.norm(C@ResolutionLU(L, U, B) - np.ravel(B))

 
        GaussChoixPivotPartiel(A, B)
        erreur3 = np.linalg.norm(A@GaussChoixPivotPartiel(A, B) - np.ravel(B))
       

        GaussChoixPivotTotal(A, B)
        erreur4 = np.linalg.norm(A@GaussChoixPivotTotal(A, B) - np.ravel(B))
 

        indices.append(i)
        y.append(erreur)
        y2.append(erreur2)
        y3.append(erreur3)
        y4.append(erreur4)
       
    plt.plot(indices, y, color='r', label="Gauss")
    plt.plot(indices, y2, color='b', label="LU")
    plt.plot(indices, y3, color='g', label="Pivot Partiel")
    plt.plot(indices, y4, color='k', label="Pivot Total")
   
    plt.title("                         Erreur de calcul en fonction de la dimension de la matrice")
   
    plt.xlabel("Dimension de la matrice")
    plt.ylabel("Erreur de calcul")
   
    plt.legend()
      
    plt.show()

Courbe_d_erreur()