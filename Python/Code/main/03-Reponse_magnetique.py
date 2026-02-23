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

# --- Importations des bibliothèques ---
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from library import Modelisation as mod
import numpy as np
import matplotlib.pyplot as plt
import logging
plt.close("all")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# --- Constantes code ---
PATH_out = "Python/Exports/03-Reponse_magnetique/"
Te = 4000 * 0.00001     # Temps d'echantillonage [s]
Ne = 4000               # Nombre d'echantillon [.]

# --- Variable ---
mu_0 = 4e-7 * np.pi         # Permeabilite magnetique du vide [H.m-1]
mu_r = 3.5                  # Permeabilite relative de l'Alnico 5 [.]
K = 0.7                     # Coeffient K [.]
x_hp1, x_hp2 = 0.147, 0.165 # Position du micro humbucker [m]
S = 1.5e-4                  # Surface de la bobine du micro [m2]
Rs_s = 5.26e3               # Resistance DC du SingleCoil [Ohm]
L_s = 2.43                   # Inductance du SingleCoil [H]
N_s = np.sqrt((L_s * 1.5e-2) / (mu_0 * mu_r * S * K))
L_hp = 1.77                  # Inductance du micro [H]
N_hp = np.sqrt(L_hp * 1.5e-2 / (mu_0 * mu_r * S * K)) / 2
L_hs = 4.71                  # Inductance du micro [H]
N_hs = np.sqrt(L_hs * 1.5e-2 / (mu_0 * mu_r * S * K)) / 2
FB = 50                     # Frequence du champ parasite [Hz]
B0 = 46 * 10**(-6)          # Champ magnetique au centre des bobines Helmholtz [T]
alpha = 4                   # Coefficient alpha [m-1]
Delta = 0.018

# --- Couleurs graphiques ---
kw_Hp_th = {'color': '#000080', 'label': 'Humbucker parallèle - Théorie', 'linewidth': 2}
kw_S_th = {'color': '#8B0000', 'label': 'SingleCoil - Théorie', 'linewidth': 2}
kw_Hs_th = {'color': '#006400', 'label': 'Humbucker série - Théorie', 'linewidth': 2}
kw_Hp_exp = {'color': '#9400D3', 'label': 'Humbucker parallèle - expérimentale', 'marker':"o", 'markersize':5, 'linestyle':'None'}
kw_Hs_exp = {'color': '#00CD00', 'label': 'Humbucker série - expérimentale', 'marker':"o", 'markersize':5, 'linestyle':'None'}
kw_S_exp = {'color': '#FF8C00', 'label': 'SingleCoil - expérimentale', 'marker':"o", 'markersize':5, 'linestyle':'None'}

# --- Données experimentales ---
List_Bpar = np.array([0, 5.5, 11, 16.5, 22, 27.5, 33, 38.5, 44, 49.5]) * 10**-6

def _read_amplitude_file(path): # Lire les fichiers contenant les données
    s_list = [0]
    hp_list = [0]
    with open(path, 'r') as fh:
        lines = fh.readlines()
    for ligne in lines[1:]:
        cols = ligne.strip().split('\t')
        cols = [c.replace('\n', '') for c in cols]
        cols = [c.replace(',', '.') for c in cols]
        label, amp_s = cols[0], float(cols[1])
        if "Humbucker" in label:
            hp_list.append(amp_s)
        elif "Single Coil" in label:
            s_list.append(amp_s)
    return np.array(hp_list), np.array(s_list)

List_amp_hp_exp, List_amp_s_exp = _read_amplitude_file("Python/data/filter/!Reponse_magnetique_amplitude.csv")

# ajustement linéaire expérimental
k_h_exp, _ = np.polyfit(List_Bpar, List_amp_hp_exp, 1)
k_s_exp, _ = np.polyfit(List_Bpar, List_amp_s_exp, 1)
List_amp_hp_exp_lin = k_h_exp * List_Bpar
List_amp_s_exp_lin = k_s_exp * List_Bpar


# --- Données Théorique ---
ks = N_s * S * 2 * np.pi * FB * mu_r
khp = 0.5 * N_hp * S * 2 * np.pi * FB * mu_r * alpha * Delta
khs = N_hs * S * 2 * np.pi * FB * mu_r * alpha * Delta

List_amp_s_th = ks * List_Bpar
List_amp_hp_th = khp * List_Bpar
List_amp_hs_th = khs * List_Bpar

# --- Parrallèle ---
CMRRp_exp = 20 * np.log10(k_s_exp / k_h_exp)
CMRRp_th = 20 * np.log10(ks / khp)
tauxp_th = ks / khp
tauxp_exp = k_s_exp / k_h_exp

# --- Série ---
#CMRRp_exp = 20*np.log10(k_s_exp / k_hs_exp)
CMRRs_th = 20 * np.log10(ks / khs)
tauxs_th = ks / khs
#tauxp_exp = k_s_exp / k_hs_exp

# --- Ecarts --- 
eps_cmrrp = abs((CMRRp_th - CMRRp_exp) / (CMRRp_th)) * 100
#eps_cmrrs = abs((CMRRs_th - CMRRs_exp)/(CMRRs_th)) *100

# --- Droite sensibilité ---
mod.save_plot(PATH_out + "Droite_de_sensibilite.png",
        (List_Bpar*1e6, List_amp_s_th*1e3, kw_S_th),
        (List_Bpar*1e6, List_amp_s_exp*1e3, kw_S_exp),
        (List_Bpar*1e6, List_amp_hp_th*1e3, kw_Hp_th),
        (List_Bpar*1e6, List_amp_hp_exp*1e3, kw_Hp_exp),
        (List_Bpar*1e6, List_amp_hs_th*1e3, kw_Hs_th),
        title="Tension induite dans le micro en fonction d'un \nchamp magnetique exterieur à 50 $Hz$",
        xlabel="Amplitude du champ parasite [$µT$]", ylabel="Amplitude Tension [$mV$]")

print(f"k_s = {ks*10**-3} mV/microTesla")
print(f"k_hp = {khp*10**-3} mV/microTesla")
print(f"k_hs = {khs*10**-3} mV/microTesla")

print(f"Taux theorique P = {tauxp_th}")
print(f"CMRR theorique P = {CMRRp_th} dB")
print(f"Taux theorique S = {tauxs_th}")
print(f"CMRR theorique S = {CMRRs_th} dB")


print("\nCalcul des sensibilites et CMRR experimental")
print(f"k_s_exp = {k_s_exp*10**-3} mV/microTesla")
print(f"k_h_exp = {k_h_exp*10**-3} mV/microTesla")
print(f"Taux exp = {tauxp_exp}")
print(f"CMRR_exp = {20*np.log10( k_s_exp / k_h_exp )} dB")
print("\nCalcul des erreurs relatives...")

print(f"Erreur relative cmrrp : {eps_cmrrp} %")
#print(f"Erreur relative cmrrs : {eps_cmrrs} %")