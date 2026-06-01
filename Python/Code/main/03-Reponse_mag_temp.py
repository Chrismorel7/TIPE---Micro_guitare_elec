"""Module de simulation temporelle des micros de guitare électrique.

Ce script compare les signaux temporels expérimentaux (oscilloscope) 
avec les modèles théoriques de Faraday pour un SingleCoil et un Humbucker.

Auteur: Christophe Morel.
Établissement: Lycée Ferdinand Buisson, CPGE PT.
Projet: TIPE N°39662 - Session 2026.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from library import Modelisation as mod

# --- Configuration de l'affichage ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.close("all")

kw_Hp_th = {'color': '#0000FF', 'label': 'Humbucker // (Théorie)', 'linewidth': 3}
kw_S_th = {'color': '#FF0000', 'label': 'SingleCoil (Théorie)', 'linewidth': 3}
kw_Hs_th = {'color': '#009900', 'label': 'Humbucker -- (Théorie)', 'linewidth': 3}
kw_Hp_exp = {'color': '#6666FF', 'label': 'Humbucker // (Expérience)', 'marker': "o", 'markersize': 2, 'linestyle': 'none'}
kw_Hs_exp = {'color': '#66CC66', 'label': 'Humbucker -- (Expérience)', 'marker': "o", 'markersize': 2, 'linestyle': 'none'}
kw_S_exp = {'color': '#FF6666', 'label': 'SingleCoil (Expérience)', 'marker': "o", 'markersize': 2, 'linestyle': 'none'}

# --- 1. Constantes physiques et géométriques ---
mu_0 = 4e-7 * np.pi         # Perméabilité magnétique du vide [H.m-1]
mu_r = 12                  # Perméabilité relative du noyau (Alnico 5)
K = 0.7                     # Coefficient de couplage
S = 1.5e-4                  # Surface moyenne d'une spire [m2]

FB = 50                     # Fréquence du champ parasite [Hz]
omega_B = 2 * np.pi * FB    # Pulsation [rad/s]
B0 = 3.68e-3                 # Amplitude du champ parasite [T] (2.6 mT)
alpha = 4                   # Gradient du champ parasite [m-1]
Delta = 0.018               # Espacement entre les bobines du Humbucker [m]

# --- 2. Calcul du nombre de spires (à partir des inductances) ---
L_s = 2.43  # [H]
N_s = np.sqrt((L_s * 1.5e-2) / (mu_0 * mu_r * S * K))

L_hs = 4.71 # [H]
N_hs = np.sqrt(L_hs * 1.5e-2 / (mu_0 * mu_r * S * K)) / 2

# --- 3. Fonctions utilitaires ---
def charger_temporel(filepath):
    """Charge les données temporelles d'un fichier CSV (Oscilloscope)."""
    try:
        # skiprows=2 pour ignorer les en-têtes texte des fichiers Keysight
        data = np.loadtxt(filepath, delimiter=',', skiprows=2)
        t = data[:, 0]
        v = data[:, 1]
        return t, v
    except Exception as e:
        print(f"Erreur lors de la lecture de {filepath} : {e}")
        return np.array([]), np.array([])

# --- 4. Chargement des données expérimentales ---
file_sc = 'Python/data/raw/03-Reponse_mag/temp_sc.csv'
file_hs = 'Python/data/raw/03-Reponse_mag/temp_hs.csv'

t_sc, v_sc_exp = charger_temporel(file_sc)
t_hs, v_hs_exp = charger_temporel(file_hs)

# --- 5. Modèles Théoriques ---
# Note : on ajoute une phase 'phi' (en radians) pour synchroniser visuellement 
# la sinusoïde théorique avec le signal de l'oscilloscope qui ne commence pas à t=0.
phi_sc = 1.57  # Valeur à ajuster manuellement selon le déclenchement de ton oscillo
phi_hs =0  # Valeur à ajuster manuellement selon le déclenchement de ton oscillo

if len(t_sc) > 0:
    v_sc_th = B0 * N_s * S * mu_r * omega_B * np.sin(omega_B * t_sc + phi_sc)

if len(t_hs) > 0:
    v_hs_th = B0 * alpha * Delta * N_hs * mu_r * S * omega_B * np.sin(omega_B * t_hs + phi_hs)

# --- 6. Tracé des graphiques ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Graphique 1 : SingleCoil
mod.save_plot("Python/Exports/03-Reponse_magnetique/Singlecoil.png", (t_sc, v_sc_th, kw_S_th), (t_sc, v_sc_exp, kw_S_exp),
            title="Réponse temporelle des micros pour un champ de 2.6mT", xlabel="temps [s]", ylabel="Tension induite [V]",
            ylim=(-14, 14))

# Graphique 2 : Humbucker Série
mod.save_plot("Python/Exports/03-Reponse_magnetique/Humbucker.png", (t_hs, v_hs_th, kw_Hs_th), (t_hs, v_hs_exp, kw_Hs_exp),
            title="Réponse temporelle des micros pour un champ de 2.6mT", xlabel="temps [s]", ylabel="Tension induite [V]",
            ylim=(-14, 14))