"""Module de simulation magnétique des micros de guitare électrique.

Ce module implémente les modèles de transduction electromagnétique pour 
comparer les performances des micros SingleCoil et Humbucker. 
Il repose sur la modélisation d'un champ magnétique exterieur parasite.

Auteur: Christophe Morel.
Établissement: Lycée Ferdinand Buisson, CPGE PT.
Projet: TIPE N°39662 - Session 2026.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os

plt.close("all")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Constantes physiques et géométriques ---
mu_0 = 4e-7 * np.pi
mu_r = 2
K = 0.7
S = 5e-4
FB = 50 
alpha = 1 
Delta = 0.018
h_bobine = 1.5e-2

n_helmholtz = 95
r_helmholtz = 0.065

L_s = 2.43
N_s = np.sqrt((L_s * h_bobine) / (mu_0 * mu_r * S * K))

L_hp = 1.77
N_hp = np.sqrt((2 * L_hp * h_bobine) / (mu_0 * mu_r * S * K))

L_hs = 4.71
N_hs = np.sqrt((L_hs * h_bobine) / (2 * mu_0 * mu_r * S * K))

# --- Fonctions utilitaires ---
def load_and_compute_B(filepath):
    """Charge les données expérimentales et calcule les amplitudes (crêtes) du champ B et de la f.e.m."""
    try:
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        
        # Extraction et conversion des incertitudes de type RMS en amplitudes
        ui = data[0, 0] * np.sqrt(2)
        usig = data[0, 2]
        
        # Incertitude sur le rayon des bobines de Helmholtz (en mètres)
        u_r = 0.005 
        
        # Extraction des mesures (lignes suivantes)
        mesures = data[1:, :]
        
        # Conversion des valeurs efficaces du fichier en amplitudes (crêtes)
        I_exp = mesures[:, 0] * np.sqrt(2)
        e_exp = mesures[:, 2]
        
        # Conversion Courant -> Champ B (B_exp sera naturellement une amplitude)
        facteur_B = (4/5)**1.5 * (mu_0 * n_helmholtz) / r_helmholtz
        B_exp = facteur_B * I_exp
        
        # Propagation des incertitudes sur l'amplitude de B
        u_B_I = facteur_B * ui  
        u_B_r = B_exp * (u_r / r_helmholtz)  
        u_B = np.sqrt(u_B_I**2 + u_B_r**2)
        
        return B_exp, e_exp, u_B, usig
    except Exception as err:
        logging.error(f"Erreur avec {filepath}: {err}")
        return np.array([]), np.array([]), 0, 0

def calc_sensibilite_exp(B, e):
    """Régression linéaire passant par l'origine pour calculer la pente k."""
    if len(B) == 0: return 0
    return np.sum(B * e) / np.sum(B**2)

# --- 1. Données expérimentales ---
PATH_DATA = "Python/data/raw/03-Reponse_mag/"
B_sc_exp, e_sc_exp, uB_sc, ue_sc = load_and_compute_B(PATH_DATA + "13-05_S.csv")
B_hp_exp, e_hp_exp, uB_hp, ue_hp = load_and_compute_B(PATH_DATA + "13-05_HP.csv")
B_hs_exp, e_hs_exp, uB_hs, ue_hs = load_and_compute_B(PATH_DATA + "13-05_HS.csv")

k_sc_exp = calc_sensibilite_exp(B_sc_exp, e_sc_exp)
k_hp_exp = calc_sensibilite_exp(B_hp_exp, e_hp_exp)
k_hs_exp = calc_sensibilite_exp(B_hs_exp, e_hs_exp)

# --- 2. Modèles Théoriques ---
k_sc_th = N_s * S * 2 * np.pi * FB * mu_r
k_hp_th = 0.5 * N_hp * S * 2 * np.pi * FB * mu_r * alpha * Delta
k_hs_th = N_hs * S * 2 * np.pi * FB * mu_r * alpha * Delta

# --- 3. Tracé des graphiques ---
B_max = max([np.max(B_sc_exp) if len(B_sc_exp)>0 else 0, 
             np.max(B_hp_exp) if len(B_hp_exp)>0 else 0, 
             np.max(B_hs_exp) if len(B_hs_exp)>0 else 0])
B_th = np.linspace(0, B_max * 1.1, 100)


plt.figure(figsize=(6, 5))

plt.plot(B_th*1e3, (k_sc_th * B_th), color='#FF0000', linewidth=3, label='SingleCoil (Théorie)')
plt.plot(B_th*1e3, (k_hs_th * B_th), color='#009900', linewidth=3, label='Humbucker -- (Théorie)')
plt.plot(B_th*1e3, (k_hp_th * B_th), color='#0000FF', linewidth=3, label='Humbucker // (Théorie)')

# Modification des points en croix (+) avec barres d'erreur
if len(B_sc_exp) > 0:
    plt.errorbar(B_sc_exp*1e3, e_sc_exp, xerr=uB_sc*1e3, yerr=ue_sc, 
                 fmt='none', color='#FF6666', label='SingleCoil (Expérience)', capsize=5)
if len(B_hs_exp) > 0:
    plt.errorbar(B_hs_exp*1e3, e_hs_exp, xerr=uB_hs*1e3, yerr=ue_hs, 
                 fmt='none', color='#66CC66', label='Humbucker -- (Expérience)', capsize=5)
if len(B_hp_exp) > 0:
    plt.errorbar(B_hp_exp*1e3, e_hp_exp, xerr=uB_hp*1e3, yerr=ue_hp, 
                 fmt='none', color='#6666FF', label='Humbucker // (Expérience)', capsize=5)

plt.title("Sensibilité des micros à un champ parasite (50 Hz)", fontsize=12, fontweight='bold')
plt.xlabel("Amplitude du champ parasite [mT]", fontsize=12)
plt.ylabel("Amplitude de la tension induite [V]", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')
plt.tight_layout()

os.makedirs("Python/Exports/03-Reponse_magnetique/", exist_ok=True)
plt.savefig("Python/Exports/03-Reponse_magnetique/Droite_de_sensibilite.png", dpi=300)

# --- 4. Affichage des grandeurs (Conversion de V/T en V/mT) ---
print(f"k_s_th : {k_sc_th*1e-3} V/mT")
print(f"k_hp_th : {k_hp_th*1e-3} V/mT")
print(f"k_hs_th : {k_hs_th*1e-3} V/mT")

print(f"k_s (Exp)  : {k_sc_exp * 1e-3:.4f} V/mT")
print(f"k_hp (Exp) : {k_hp_exp * 1e-3:.4f} V/mT")
print(f"k_hs (Exp) : {k_hs_exp * 1e-3:.4f} V/mT")
if k_hp_exp != 0:
    print(f"CMRR // :\tExp: {20 * np.log10(k_sc_exp / k_hp_exp):.1f} dB\tTH: {20 * np.log10(k_sc_th / k_hp_th)} dB")
if k_hs_exp != 0:
    print(f"CMRR -- :\tExp: {20 * np.log10(k_sc_exp / k_hs_exp):.1f} dB\tTH: {20 * np.log10(k_sc_th / k_hs_th)} dB")