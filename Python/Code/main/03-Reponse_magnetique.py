"""Modèle electrique du micro de guitare electrique et affichage de la reponse en frequence via un diagramme de Bode.
    """

print("\n\n---\nModele_elec.py\n---\n")

                                            #!####################!#
                                            #!   Bibliothèques    !#
                                            #!####################!#

print("\nImportation des bibliotheques...")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from library import Modelisation as mod
import numpy as np
import matplotlib.pyplot as plt
import logging
plt.close("all")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
print("Bibliotheques importees.")

#! ------------------ Constantes ---------------------------------
print("-" * 100)
print("\nInitialisation des constantes")
#* Constantes code
PATH_out = "Python/Exports/03-Reponse_magnetique/"
#* Constante physiqu
rho = 5.41   # resistance lineique du cuivre (Ohm.m-1)
mu0 = 4e-7 * np.pi      # Permeabilite magnetique du vide (H.m-1)
mu_r = 3.5
K = 0.7
#* Parametre Micros
#* Parametre maquette
xm = 0.105              # Position du micro singlecoil
x1 = 0.140              # Position du micro humbucker 1
x2 = 0.160              # Position du micro humbucker 2
L = 0.845               # Longueur de la corde
mu = 9 * 10**(-3)       # Masse lineique de la corde
dBdy = 0.03           #? Gradient du champ magnetique (A verifier ???)
#* Parametre Micros
S = 1.5e-4        #? Surface de la bobine du micro
#* Parametre Micro Singlecoil
RCC_s = 5.3e3     # Resistance DC micro singlecoil
p_s = 0.13
long_s = RCC_s / rho
l_s = 1.5e-2             #? Epaisseur du micro
L_s = 2.4
N_s = np.sqrt(L_s * l_s / (mu0 * mu_r * S * K))
#* Parametre Micro Humbucker
RCC_h = 4.52 * 10**3    # Resistance DC micro humbucker
p_h = 0.13
long_h = RCC_h / rho
l_h = l_s
L_h = 1.84               # Inductance du micro
N_h = np.sqrt(2*L_h * l_h / (mu0 * mu_r * S * K))
#* Parametre Champ parasite
fB = 50                 # Frequence du champ parasite
B0 = 46 * 10**(-6)      # Champ magnetique au centre des bobines Helmholtz
alpha = 2               # Coefficient alpha
#* Couleurs graphiques
kw_Hp_th = {'color': '#000080', 'label': 'Humbucker parallèle - Théorie', 'linewidth': 2}
kw_S_th = {'color': '#8B0000', 'label': 'SingleCoil - Théorie', 'linewidth': 2}
kw_Hs_th = {'color': '#006400', 'label': 'Humbucker série - Théorie', 'linewidth': 2}
kw_Hp_exp = {'color': '#9400D3', 'label': 'Humbucker parallèle - experience', 'marker':"o", 'markersize':5, 'linestyle':'None'}
kw_Hs_exp = {'color': '#00CD00', 'label': 'Humbucker série - experience', 'marker':"o", 'markersize':5, 'linestyle':'None'}
kw_S_exp = {'color': '#FF8C00', 'label': 'SingleCoil - experience', 'marker':"o", 'markersize':5, 'linestyle':'None'}

if RCC_h is None or l_h is None or N_h is None:
    logging.warning("Parametres humbucker (RCC_h, l_h, N_h) non defini(s). Certaines simulations pourront utiliser N_s en substitution.")

print("Constantes initialisees.")
print("\nValeurs des parametres (avec unites) :")

params = [
    ("rho", rho, "Ohm.m", "Resistivite du fil de cuivre"),
    ("S", S, "m2", "Surface de la bobine"),
    ("N_s", N_s, "spires", "Nombre de spires du micro singlecoil"),
    ("N_h", N_h, "spires", "Nombre de spires micro humbucker"),
    ("fB", fB, "Hz", "Frequence du champ parasite"),
    ("B0", B0, "T", "Champ magnetique au centre"),
    ("alpha", alpha, "-", "Coefficient de rejet"),
]

try :
    print(f"{'Var':<8} {'Valeur':>18} {'Unite':<10} {'Description':<45}")
    print("-" * 85)
    for i, (var, val, unit, desc) in enumerate(params):
        if isinstance(val, (int, np.integer)):
            val_str = f"{val:d}"
        elif isinstance(val, (float, np.floating)):
            val_str = f"{val:.6g}"
        else:
            val_str = str(val)
        if i == 0:
            print("-" * 70)
        print(f"{var:<8} {val_str:>18} {unit:<10} {desc:<45}")
except Exception as e:
    logging.error(f"Erreur lors de l'affichage des parametres : {e}")

print("-" * 85)


# ======================== Resultats / Execution ========================
print("Lecture des donnees de test...")
List_Bpar = np.array([0, 5.5, 11, 16.5, 22, 27.5, 33, 38.5, 44, 49.5]) * 10**-6
ks = mu_r*N_s*S*2*np.pi*fB
kh = N_h*S*2*np.pi*fB*alpha*np.abs(x1-x2)
List_amp_h_th = np.array([kh * B for B in List_Bpar])
List_amp_s_th = np.array([ks * B for B in List_Bpar])

List_amp_h_test = [0]
List_amp_s_test = [0]

file = open("Python/data/filter/!Reponse_magnetique_amplitude.csv")
L_lignes = file.readlines()
for ligne in L_lignes[1:]:
    ligne = ligne.strip().split('\t')
    ligne = [x.replace('\n', '') for x in ligne]
    ligne = [x.replace(',', '.') for x in ligne]
    label = ligne[0]
    amp = float(ligne[1])
    if "Humbucker" in label:
        List_amp_h_test.append(amp - 0.005)
    elif "Single Coil" in label:
        List_amp_s_test.append(amp)

List_amp_h_test = np.array(List_amp_h_test)
List_amp_s_test = np.array(List_amp_s_test)

k_h_exp, _ = np.polyfit(List_Bpar, List_amp_h_test, 1)
k_s_exp, _ = np.polyfit(List_Bpar, List_amp_s_test, 1)
List_amp_h_test_lin = k_h_exp * List_Bpar
List_amp_s_test_lin = k_s_exp * List_Bpar


doc_path = "Python/data/filter/SingleCoil_par_17,5V.csv"
try:
    with open(doc_path) as doc:
        L_lignes = doc.readlines()
    Test_temps_s_p = []
    Test_s_p = []
    for i in range(10001, 35001):
        values = L_lignes[i].strip().split("\t")
        values = [float(v.replace("\n", "")) for v in values]
        Test_temps_s_p.append(values[0])
        Test_s_p.append(values[1])
    Test_temps_s_p = np.array(Test_temps_s_p)
    Test_temps_s_p -= Test_temps_s_p[0]
    Test_s_p = np.array(Test_s_p)
    print("Donnees de test Single Coil parasite seules lues.")
except Exception as e:
    logging.error(f"Erreur lors de la lecture des donnees de test Single Coil parasite seules : {e}")


doc_path = "Python/data/filter/Humbucker_par_8V.csv"
try:
    with open(doc_path) as doc:
        L_lignes = doc.readlines()
    Test_temps_h_p = []
    Test_h_p = []
    for i in range(1, 25001):
        values = L_lignes[i].strip().split("\t")
        values = [float(v.replace("\n", "")) for v in values]
        Test_temps_h_p.append(values[0])
        Test_h_p.append(values[1])
    Test_temps_h_p = np.array(Test_temps_h_p)
    Test_temps_h_p -= Test_temps_h_p[0]
    Test_h_p = np.array(Test_h_p)
    print("Donnees de test Humbucker parasite seules lues.")
except Exception as e:
    logging.error(f"Erreur lors de la lecture des donnees de test Humbucker parasite seules : {e}")


print("\nCalcul des tensions induites au champ parasite...")
Te = 4000 * 0.00001
Ne = 4000
temps = np.linspace(0, Te, Ne, endpoint=False)
fem_s_par = mod.fem_par_s(N_s, fB, temps, B0, S, 2*np.pi/3)
fem_h_par = mod.fem_par_h(N_h, x1, x2, fB, temps, B0, alpha, S)


#* --- Micro SingleCoil (Parasite) ---*#
mod.save_plot(PATH_out + "Singlecoil.png",
        (temps, Test_s_p[:4000], kw_S_exp),
        (temps, fem_s_par, kw_S_th),
        title="Tension induite dans le micro SingleCoil \nsoumis à un champs magnetique de 50µT",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]", ylim=(-0.2, 0.2))
print("Courbes sauvegardees sous Python/Exports/03-Parasite_singlecoil.png.")
#* --- Micro Humbucker (Parasite) ---*#
mod.save_plot(PATH_out + "Humbucker.png",
        (temps, Test_h_p[:4000], kw_Hp_exp),
        (temps, fem_h_par, kw_Hp_th),
        title="Tension induite dans le micro Humbucker \nsoumis à un champs magnetique de 50µT",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]", ylim=(-0.2, 0.2))
print("Courbes sauvegardees sous Python/Exports/03-Parasite_humbucker.png.")
#* --- Droite champ parasite ---*#
mod.save_plot(PATH_out + "Droite_de_sensibilite.png",
        (List_Bpar*10**6, List_amp_h_th*10**3, kw_Hp_th),
        (List_Bpar*10**6, List_amp_h_test*10**3, kw_Hp_exp),
        (List_Bpar*10**6, List_amp_h_test_lin*10**3, {'linestyle': 'dashed', 'color': '#9400D3'}),
        (List_Bpar*10**6, List_amp_s_th*10**3, kw_S_th),
        (List_Bpar*10**6, List_amp_s_test*10**3, kw_S_exp),
        (List_Bpar*10**6, List_amp_s_test_lin*10**3, {'linestyle': 'dashed', 'color': '#FF8C00'}),
        title="Tension induite dans le micro en fonction d'un \nchamp magnetique exterieur à 50 $Hz$",
        xlabel="Amplitude du champ parasite [$µT$]", ylabel="Ampltitude Tension [$mV$]")
print("Courbes sauvegardees sous Python/Exports/03-Droite.png.")


CMRR_exp = 20*np.log10(k_s_exp / k_h_exp)
CMRR_th = 20*np.log10(ks/kh)
taux_th = ks/kh
taux_exp = k_s_exp / k_h_exp

eps_h = abs((k_h_exp - kh) / kh) * 100
eps_s = abs((k_s_exp - ks) / ks) * 100
eps = abs((taux_th - taux_exp) / (taux_th)) * 100
eps_cmrr = abs((CMRR_th - CMRR_exp)/(CMRR_th)) *100

print("\nCalcul des sensibilites et CMRR theorique")
print(f"k_s = {ks*10**-3} mV/microTesla")
print(f"k_h = {kh*10**-3} mV/microTesla")
print(f"Taux theorique = {taux_th}")
print(f"CMRR theorique = {20*np.log10(ks / kh)} dB")

print("\nCalcul des sensibilites et CMRR experimental")
print(f"k_s_exp = {k_s_exp*10**-3} mV/microTesla")
print(f"k_h_exp = {k_h_exp*10**-3} mV/microTesla")
print(f"Taux exp = {taux_exp}")
print(f"CMRR_exp = {20*np.log10( k_s_exp / k_h_exp )} dB")
print("\nCalcul des erreurs relatives...")

print(f"\nErreur relative k_h : {eps_h} %")
print(f"Erreur relative k_s : {eps_s} %")
print(f"Erreur relative taux : {eps} %")
print(f"Erreur relative cmrr : {eps_cmrr} %")

print("-" * 100)
print("\n---\nFin de l'execution de Modele_fem.py\n---\n")