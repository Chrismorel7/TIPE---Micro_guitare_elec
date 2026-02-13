"""Module de simulation magnétique des micros de guitare électrique.

Ce module implémente les modèles de transduction electromagnétique pour 
comparer les performances des micros SingleCoil et Humbucker. 
Il repose sur la modélisation d'un champ magnétique exterieur parasite.

Calculs principaux :
    - Force électromotrice e(x, t) induite par Faraday
    - Taux de rejection de mode commun CMRR
    
Auteur: Christophe Morel.
Établissement: Lycée Ferdinand Buisson, CPGE PT.
Projet: TIPE N°39662 - Session 2026.
"""

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

                                            #!####################!#
                                            #!    Constantes      !#
                                            #!####################!#

print("\nInitialisation des constantes")
# --- Constantes code ---
PATH_out = "Python/Exports/03-Reponse_magnetique/"

# --- Variable ---
mu_0 = 4e-7 * np.pi         # Permeabilite magnetique du vide [H.m-1]
mu_r = 3.5                  # Permeabilite relative de l'Alnico 5 [.]
K = 0.7                     # Coeffient K [.]
x_hp1, x_hp2 = 0.147, 0.165 # Position du micro humbucker [m]
S = 1.5e-4                  # Surface de la bobine du micro [m2]
Rs_s = 5.26e3               # Resistance DC du SingleCoil [Ohm]
L_s = 2.4                   # Inductance du SingleCoil [H]
N_s = np.sqrt((L_s * 1.5e-2) / (mu_0 * mu_r * S * K))
L_h = 1.84                  # Inductance du micro [H]
N_hp = np.sqrt(2*L_h * 1.5e-2 / (mu_0 * mu_r * S * K))
FB = 50                     # Frequence du champ parasite [Hz]
B0 = 46 * 10**(-6)          # Champ magnetique au centre des bobines Helmholtz [T]
alpha = 4                   # Coefficient alpha [m-1]

# --- Couleurs graphiques ---
kw_Hp_th = {'color': '#000080', 'label': 'Humbucker parallèle - Théorie', 'linewidth': 2}
kw_S_th = {'color': '#8B0000', 'label': 'SingleCoil - Théorie', 'linewidth': 2}
kw_Hs_th = {'color': '#006400', 'label': 'Humbucker série - Théorie', 'linewidth': 2}
kw_Hp_exp = {'color': '#9400D3', 'label': 'Humbucker parallèle - experience', 'marker':"o", 'markersize':5, 'linestyle':'None'}
kw_Hs_exp = {'color': '#00CD00', 'label': 'Humbucker série - experience', 'marker':"o", 'markersize':5, 'linestyle':'None'}
kw_S_exp = {'color': '#FF8C00', 'label': 'SingleCoil - experience', 'marker':"o", 'markersize':5, 'linestyle':'None'}

print("Constantes initialisees.")

                                            #!####################!#
                                            #!      Get data      !#
                                            #!####################!#

print("Lecture des donnees de test...")
List_Bpar = np.array([0, 5.5, 11, 16.5, 22, 27.5, 33, 38.5, 44, 49.5]) * 10**-6
ks = N_s*S*2*np.pi*FB*mu_r
kh = N_hp*S*2*np.pi*FB*alpha*np.abs(x_hp1-x_hp2)*mu_r
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
fem_s_par = mod.fem_par_s(temps, B0, np.pi / 2)
fem_h_par = mod.fem_par_h(N_hp, x_hp1, x_hp2, temps, B0, alpha)

CMRR_exp = 20*np.log10(k_s_exp / k_h_exp)
CMRR_th = 20*np.log10(ks/kh)
taux_th = ks/kh
taux_exp = k_s_exp / k_h_exp

eps_h = abs((k_h_exp - kh) / kh) * 100
eps_s = abs((k_s_exp - ks) / ks) * 100
eps = abs((taux_th - taux_exp) / (taux_th)) * 100
eps_cmrr = abs((CMRR_th - CMRR_exp)/(CMRR_th)) *100

                                            #!####################!#
                                            #!       Exports      !#
                                            #!####################!#

# --- Micro SingleCoil ---
mod.save_plot(PATH_out + "Singlecoil.png",
        (temps, Test_s_p[:4000], kw_S_exp),
        (temps, fem_s_par, kw_S_th),
        title="Tension induite dans le micro SingleCoil \nsoumis à un champs magnetique de 50µT",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]", ylim=(-0.2, 0.2))

#--- Micro Humbucker (Parasite) ---
mod.save_plot(PATH_out + "Humbucker.png",
        (temps, Test_h_p[:4000], kw_Hp_exp),
        (temps, fem_h_par, kw_Hp_th),
        title="Tension induite dans le micro Humbucker \nsoumis à un champs magnetique de 50µT",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]", ylim=(-0.2, 0.2))

# --- Droite sensibilité ---
mod.save_plot(PATH_out + "Droite_de_sensibilite.png",
        (List_Bpar*10**6, List_amp_h_th*10**3, kw_Hp_th),
        (List_Bpar*10**6, List_amp_h_test*10**3, kw_Hp_exp),
        (List_Bpar*10**6, List_amp_h_test_lin*10**3, {'linestyle': 'dashed', 'color': '#9400D3'}),
        (List_Bpar*10**6, List_amp_s_th*10**3, kw_S_th),
        (List_Bpar*10**6, List_amp_s_test*10**3, kw_S_exp),
        (List_Bpar*10**6, List_amp_s_test_lin*10**3, {'linestyle': 'dashed', 'color': '#FF8C00'}),
        title="Tension induite dans le micro en fonction d'un \nchamp magnetique exterieur à 50 $Hz$",
        xlabel="Amplitude du champ parasite [$µT$]", ylabel="Ampltitude Tension [$mV$]")


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

print("\n---\nFin de l'execution de Modele_fem.py\n---\n")