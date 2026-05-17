"""Module de simulation acoustique des micros de guitare électrique.

Ce module implémente les modèles de transduction électromagnétique pour 
comparer les performances des micros SingleCoil et Humbucker. 
Il repose sur la modélisation de l'induction par une corde vibrante.

Auteur: Christophe Morel.
Établissement: Lycée Ferdinand Buisson, CPGE PT.
Projet: TIPE N°39662 - Session 2026.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from library import Modelisation as mod
import numpy as np
import matplotlib.pyplot as plt
import logging

plt.close("all")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration globale ---
tracer_fft = True
PATH_exp = "Python/Exports/02-Reponse_mecanique/"
Na = 20          # Nombre de modes considérés
Te = 1           # Temps d'échantillonnage [s]
Ne = 10000       # Nombre de points d'échantillonnage
fs = Ne / Te     # Fréquence d'échantillonnage [Hz]
yerr_temp = (1e-5 / 2)  # Incertitude de mesure oscillo [V]
prominence_val = 0.03   # Seuil de proéminence uniforme et symétrique pour la détection des pics

# --- Constantes physiques et géométriques ---
mu_0 = 4e-7 * np.pi          # Perméabilité magnétique du vide [H.m-1]
mu_r = 3.5                   # Perméabilité relative de l'Alnico 5 [.]
K = 0.7                      # Coefficient de forme [.]
S = 1.5e-4                   # Surface d'une bobine [m2]
h_bobine = 1.5e-2            # Hauteur de la bobine [m]
x_hp1, x_hp2 = 0.147, 0.165  # Position du humbucker parallèle [m]
x_hs1, x_hs2 = 0.202, 0.220  # Position du humbucker série [m]
A0 = 1e-3                    # Amplitude initiale de vibration [m]
x0 = 0.300                   # Origine de la perturbation [m]
F = 80                       # Fréquence fondamentale [Hz]

# --- Calcul rigoureux du nombre de spires par bobine ---
L_s = 2.43                   
N_s = np.sqrt((L_s * h_bobine) / (mu_0 * mu_r * S * K))

L_hp = 1.77                  # Association parallèle : L_bobine = 2 * L_hp
N_hp = np.sqrt((2 * L_hp * h_bobine) / (mu_0 * mu_r * S * K))

L_hs = 4.71                  # Association série : L_bobine = L_hs / 2
N_hs = np.sqrt((L_hs * h_bobine) / (2 * mu_0 * mu_r * S * K))

# --- Couleurs graphiques ---
kw_Hp_th = {'color': '#0000FF', 'label': 'Humbucker // (Théorie)', 'linewidth': 3}
kw_S_th = {'color': '#FF0000', 'label': 'SingleCoil (Théorie)', 'linewidth': 3}
kw_Hs_th = {'color': '#009900', 'label': 'Humbucker -- (Théorie)', 'linewidth': 3}
kw_Hp_exp = {'color': '#6666FF', 'label': 'Humbucker // (Expérience)', 'marker': "o", 'markersize': 8, 'linestyle': ':'}
kw_Hs_exp = {'color': '#66CC66', 'label': 'Humbucker -- (Expérience)', 'marker': "o", 'markersize': 8, 'linestyle': ':'}
kw_S_exp = {'color': '#FF6666', 'label': 'SingleCoil (Expérience)', 'marker': "o", 'markersize': 8, 'linestyle': ':'}

Title = "Spectre de la tension induite dans le micro"

# --- Lecture des données expérimentales ---
def lire_fichier(PATH):
    List_temps = []
    List_sig = []
    with open(PATH) as file:
        L_lignes = file.readlines()
    for ligne in L_lignes[2:]:
        ligne = ligne.strip().split(',')
        ligne = [x.replace('\n', '') for x in ligne]
        ligne = [x.replace('E', 'e') for x in ligne]
        List_temps.append(float(ligne[0]))
        List_sig.append(float(ligne[1]))
    return np.array(List_temps), np.array(List_sig)

temps_s, V_s_exp = lire_fichier("Python/data/raw/02-Reponse_meca/S2.csv")
temps_hp, V_hp_exp = lire_fichier("Python/data/raw/02-Reponse_meca/Hp2.csv")
temps_hs, V_hs_exp = lire_fichier("Python/data/raw/02-Reponse_meca/Hs3.csv")

# --- Calcul des FFT théoriques (suppression du /2 arbitraire inutile avec la normalisation) ---
temps = np.linspace(-Te/2, Te/2, Ne)
V_s_th = mod.fem_singlecoil(Na, temps, A0, x0)
V_hp_th = mod.fem_humbucker(Na, N_hp, x_hp1, x_hp2, temps, A0, x0)
V_hs_th = mod.fem_humbucker(Na, N_hs, x_hs1, x_hs2, temps, A0, x0)

fft_s_th = mod.compute_and_normalize_fft(V_s_th, Ne)
fft_hp_th = mod.compute_and_normalize_fft(V_hp_th, Ne)
fft_hs_th = mod.compute_and_normalize_fft(V_hs_th, Ne)

Ne_s, Ne_hp, Ne_hs = len(V_s_exp), len(V_hp_exp), len(V_hs_exp)

fft_s_exp_abs = mod.compute_and_normalize_fft(V_s_exp, Ne_s)
fft_hp_exp_abs = mod.compute_and_normalize_fft(V_hp_exp, Ne_hp)
fft_hs_exp_abs = mod.compute_and_normalize_fft(V_hs_exp, Ne_hs)

yerr_s_fft_abs = yerr_temp / np.sqrt(Ne_s)
yerr_hp_fft_abs = yerr_temp / np.sqrt(Ne_hp)
yerr_hs_fft_abs = yerr_temp / np.sqrt(Ne_hs)

freqs_th = np.fft.fftfreq(Ne, 1 / (len(V_s_th) / (temps[-1] - temps[0])))
freqs_s_exp = np.fft.fftfreq(Ne_s, 1 / (len(V_s_exp) / (temps_s[-1] - temps_s[0])))
freqs_hp_exp = np.fft.fftfreq(Ne_hp, 1 / (len(V_hp_exp) / (temps_hp[-1] - temps_hp[0])))
freqs_hs_exp = np.fft.fftfreq(Ne_hs, 1 / (len(V_hs_exp) / (temps_hs[-1] - temps_hs[0])))

mask_th = (40 < freqs_th) & (freqs_th <= 800)
mask_s = (40 < freqs_s_exp) & (freqs_s_exp <= 800)
mask_hp = (40 < freqs_hp_exp) & (freqs_hp_exp <= 800)
mask_hs = (40 < freqs_hs_exp) & (freqs_hs_exp <= 800)

freqs_th = freqs_th[mask_th]
fft_s_th = mod.normalize_signal(fft_s_th[mask_th])
fft_hp_th = mod.normalize_signal(fft_hp_th[mask_th])
fft_hs_th = mod.normalize_signal(fft_hs_th[mask_th])

freqs_env = np.linspace(80, 800, 800)
envelope = (F / freqs_env) ** 1
kw_env = {'color': 'k', 'linestyle': '--', 'label': 'Enveloppe $1/n$'}

# Double normalisation acceptée (focus exclusif sur l'enveloppe spectrale)
max_fft_s, max_fft_hp, max_fft_hs = np.max(fft_s_exp_abs), np.max(fft_hp_exp_abs), np.max(fft_hs_exp_abs)

freqs_s_exp = freqs_s_exp[mask_s]
fft_s_exp = fft_s_exp_abs[mask_s] / max_fft_s 
yerr_s_exp = np.full_like(freqs_s_exp, yerr_s_fft_abs) / max_fft_s

freqs_hp_exp = freqs_hp_exp[mask_hp]
fft_hp_exp = fft_hp_exp_abs[mask_hp] / max_fft_hp
yerr_hp_exp = np.full_like(freqs_hp_exp, yerr_hp_fft_abs) / max_fft_hp

freqs_hs_exp = freqs_hs_exp[mask_hs]
fft_hs_exp = fft_hs_exp_abs[mask_hs] / max_fft_hs
yerr_hs_exp = np.full_like(freqs_hs_exp, yerr_hs_fft_abs) / max_fft_hs

mod.save_plot(PATH_exp + "FFT/Singlecoil_th.png",
        (freqs_th, fft_s_th, kw_S_th),
        (freqs_env, envelope, kw_env),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
mod.save_plot(PATH_exp + "FFT/Singlecoil_exp.png",
        (freqs_s_exp, fft_s_exp, {**kw_S_exp, 'yerr': yerr_s_exp}),
        (freqs_env, envelope, kw_env),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
mod.save_plot(PATH_exp + "FFT/Singlecoil.png",
        (freqs_th, fft_s_th, kw_S_th),
        (freqs_s_exp, fft_s_exp, {**kw_S_exp, 'yerr': yerr_s_exp}),
        (freqs_env, envelope, kw_env),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Singlecoil.png"}")


mod.save_plot(PATH_exp + "FFT/Humbucker_parra_th.png",
        (freqs_th, fft_hp_th, kw_Hp_th),
        (freqs_env, envelope, kw_env),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
mod.save_plot(PATH_exp + "FFT/Humbucker_parra_exp.png",
        (freqs_hp_exp, fft_hp_exp, {**kw_Hp_exp, 'yerr': yerr_hp_exp}),
        (freqs_env, envelope, kw_env),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
mod.save_plot(PATH_exp + "FFT/Humbucker_parra.png",
        (freqs_th, fft_hp_th, kw_Hp_th),
        (freqs_hp_exp, fft_hp_exp, {**kw_Hp_exp, 'yerr': yerr_hp_exp}),
        (freqs_env, envelope, kw_env),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Humbucker.png"}")

mod.save_plot(PATH_exp + "FFT/Humbucker_serie_th.png",
        (freqs_th, fft_hs_th, kw_Hs_th),
        (freqs_env, envelope, kw_env),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
mod.save_plot(PATH_exp + "FFT/Humbucker_serie_exp.png",
        (freqs_hs_exp, fft_hs_exp, {**kw_Hs_exp, 'yerr': yerr_hs_exp}),
        (freqs_env, envelope, kw_env),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
mod.save_plot(PATH_exp + "FFT/Humbucker_serie.png",
        (freqs_th, fft_hs_th, kw_Hs_th),
        (freqs_hs_exp, fft_hs_exp, {**kw_Hs_exp, 'yerr': yerr_hs_exp}),
        (freqs_env, envelope, kw_env),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Humbucker.png"}")

# --- Mise en evidence du filtrage spatial ---
L_freqs_s, L_pic_s_exp = mod.extraire_pics_separes(freqs_s_exp, fft_s_exp, prominence=prominence_val, distance=5)
L_freqs_hs, L_pic_hs_exp = mod.extraire_pics_separes(freqs_hs_exp, fft_hs_exp, prominence=prominence_val, distance=5)

kw_peigne = {'color': "#000000", 'label': 'Peigne spatial', 'linewidth': 3}
kw_att = {'color': "#888888", 'label': 'Gain relatif', 'linewidth': 1}

freqs = np.linspace(30, 20000, 1000)
signal_hs, peigne_hs, signal_sc = mod.attenuation_theorique_db(freqs, x_hs1, x_hs2)

# Sécurité mathématique pour la division membre à membre des tableaux expérimentaux
min_len = min(len(L_pic_hs_exp), len(L_pic_s_exp))
L_freqs_clipped = L_freqs_hs[:min_len]

Attenuation_th = 20 * np.log10(np.abs(signal_hs / (signal_sc + 1e-6)) + 1e-6)
Attenuation_exp = 20 * np.log10(np.abs(L_pic_hs_exp[:min_len] / (L_pic_s_exp[:min_len] + 1e-6)) + 1e-6)

mod.save_plot(PATH_exp + "Filtrage_spatial.png",
        (freqs, Attenuation_th, kw_att),
        (L_freqs_clipped, Attenuation_exp, kw_Hs_exp),
        (freqs, 20 * np.log10(np.abs(peigne_hs) + 1e-6), kw_peigne),
        title="Filtrage spatial du micro Humbucker série", 
        xlabel="Fréquence [$Hz$]", ylabel="Gain [dB]", xlim=(30, 20000), vspan=True, log=True)