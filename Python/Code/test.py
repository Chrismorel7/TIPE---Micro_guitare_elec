import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
#! ------------------ Constantes ---------------------------------
print("-" * 100)
print("\nInitialisation des constantes")
#* Constante physique
rho = 1.68 * 10**(-8)   # Masse volumique du cuivre (Ohm.m)
mu0 = 4e-7 * np.pi      # Permeabilite magnetique du vide (H.m-1)
eps = 1*10**(-2)
window = 20
#* Parametre maquette
xm = 0.105              # Position du micro singlecoil
x1 = 0.140              # Position du micro humbucker 1
x2 = 0.160              # Position du micro humbucker 2
L = 0.845               # Longueur de la corde
mu = 9 * 10**(-3)       # Masse lineique de la corde
dBdy = 0.03             #? Gradient du champ magnetique (A verifier ???)
#* Parametre Micros
d = 0.063 * 10**(-3)    # Diametre du fil de cuivre
S = 7 * 10**(-4)        # Surface de la bobine du micro
#* Parametre Micro Singlecoil
RCC_s = 5.3 * 10**3     # Resistance DC micro singlecoil
lmoy_s = 0.147          #? Longueur moyenne du fil enroule
N_s = int(RCC_s * np.pi * d**2 / (4 * rho * lmoy_s))    # Nombre de spires du micro singlecoil
#* Parametre Micro Humbucker
RCC_h = 4.52 * 10**3    # Resistance DC micro humbucker
lmoy_h = lmoy_s         #? Longueur moyenne d'une bobine humbucker
N_h = 2*int(RCC_h * np.pi * d**2 / (8 * rho * lmoy_h))
#* Parametre Champ parasite
Nbob = 95               # Nombre de spires bobine Helmholtz
R = 0.130 / 2           # Rayon des bobines Helmholtz
Lbob = 1                #? Inductance bobines Helmholtz
Rbob = 10               #? Resistance bobines Helmholtz
fB = 50                 # Frequence du champ parasite
U0 = 20                 # Tension bobines Helmholtz
I0 = 5                  #? Courant bobines Helmholtz
B0 = 46 * 10**(-6)        # Champ magnetique au centre des bobines Helmholtz
alpha = 2               #? Coefficient alpha
#* Parametre variable
A = 15 * 10**(-4)          #? Amplitude initiale de vibration
x0 = 0.140                 #? Origine de la perturbation
F = 78.0                   #? Frequence de la note jouee
T = 4 * mu * (L * F)**2    #? Tension de la corde
#* Couleurs graphiques
Color_c = "#505050"
Color_h_test = "#4C9E31"
Color_s_test = "#C63032"
Color_h = "#3E77B2"
Color_s = "#EF812C"

temps = np.linspace(0, 0.04, 1000)

def fem_singlecoil(n: int, N: int, xm: float) -> np.ndarray:
    s = 0
    for i in range(1, n + 1):
        fn = (i / (2 * L)) * np.sqrt(T / mu)
        An = 2 * np.pi * fn * np.sin(i * np.pi * xm / L)
        Bn = (2 * A * np.sin((i * np.pi * x0) / (L))) / ((x0) / (L) * (1 - (x0) / (L)) * np.pi**2 * i**2)
        s += An * Bn
    return N * S * dBdy * s

def fem_humbucker(n: int, N: int, x1: float, x2: float) -> np.ndarray:
    s = 0
    for i in range(1, n + 1):
        fn = (i / (2 * L)) * np.sqrt(T / mu)
        An = 2 * np.pi * fn * (np.sin(i * np.pi * x1 / L) + np.sin(i * np.pi * x2 / L))
        Bn = (2 * A * np.sin((i * np.pi * x0) / (L))) / ((x0) / (L) * (1 - (x0) / (L)) * np.pi**2 * i**2)
        s += An * Bn
    return N * S * dBdy * s

def entree(n: int, x: float) -> np.ndarray:
    s = 0
    v = np.sqrt(T / mu)
    for i in range(1, n + 1):
        Bn = (2 * A * np.sin((i * np.pi * x0) / (L))) / ((x0) / (L) * (1 - (x0) / (L)) * np.pi**2 * i**2)
        s += Bn * np.sin(i * np.pi * x / L)
    return s

L_xm = np.linspace(0, 20, 1000)
delta = 0.10
L_amps = []
L_amph = []
L_ampe = []

for x in L_xm:
    L_amps.append(fem_singlecoil(10, N_s, x))
    L_amph.append(fem_humbucker(10, N_h, x-delta, x+delta))
    L_ampe.append(entree(10, x))

plt.plot(L_xm, L_amps)
plt.plot(L_xm, L_amph)
plt.show()