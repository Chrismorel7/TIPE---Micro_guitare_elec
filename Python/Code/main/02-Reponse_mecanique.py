"""Module de simulation acoustique des micros de guitare électrique.

Ce module implémente les modèles de transduction electromagnétique pour 
comparer les performances des micros SingleCoil et Humbucker. 
Il repose sur la modélisation de l'induction par une corde vibrante.

Calculs principaux :
    - Flux magnétique Phi(z, t) selon l'approximation dipolaire.
    - Force électromotrice e(x, t) induite par Faraday
    
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
print("Bibliotheques importees.")


# --- Constante pour code ---
tracer_fft = True
PATH_exp = "Python/Exports/02-Reponse_mecanique/"
Na = 20         # Nombre de mode considérés
Te = 1          # Temps d'echentillonage [s]
Ne = 10000      # Nombre de point d'echantillonage
fs = Ne / Te    # Frequence d'echantillonage [Hz]
yerr_temp = (1e-5 / 2 )         # Incertitude de mesure oscillo [V]

# --- Variables ---
mu_0 = 4e-7 * np.pi         # Permeabilite magnetique du vide [H.m-1]
mu_r = 3.5                  # Permeabilite relative de l'Alnico 5 [.]
K=0.7                       # Coeffient K [.]
x_hp1, x_hp2 = 0.147, 0.165 # Position du humbucker parallèle [m]
x_hs1, x_hs2 = 0.202, 0.220 # Position du humbucker série [m]
S = 1.5e-4                  # Surface d'une bobine [m2]
Rs_s = 5.26e3               # Resistance DC du SingleCoil [Ohm]
L_s = 2.43                   # Inductance du SingleCoil [H]
N_s = np.sqrt(L_s * 1.5e-2 / (mu_0 * mu_r * S * K))
L_hp = 1.77                 # Inductance du Humbucker parallèle [H]
N_hp = np.sqrt(2*L_hp * 1.5e-2 / (mu_0 * mu_r * S * K))
L_hs = 4.71                 # Inductance du Humbucker parallèle [H]
N_hs = np.sqrt(2*L_hs * 1.5e-2 / (mu_0 * mu_r * S * K))
A0 = 1e-3                   # Amplitude initiale de vibration [m]
x0 = 0.300                  # Origine de la perturbation [m]
F = 80                      # Frequence fondamentale [Hz]

# --- Couleurs graphiques ---
kw_Hp_th = {'color': '#000080', 'label': 'Humbucker parallèle - Théorie', 'linewidth': 2}
kw_S_th = {'color': '#8B0000', 'label': 'SingleCoil - Théorie', 'linewidth': 2}
kw_Hs_th = {'color': '#006400', 'label': 'Humbucker série - Théorie', 'linewidth': 2}
kw_Hp_exp = {'color': '#9400D3', 'label': 'Humbucker parallèle - expérimentale', 'marker':"o", 'markersize':5, 'linestyle':':'}
kw_Hs_exp = {'color': "#00AB00", 'label': 'Humbucker série - expérimentale', 'marker':"o", 'markersize':5, 'linestyle':':'}
kw_S_exp = {'color': '#FF8C00', 'label': 'SingleCoil - expérimentale', 'marker':"o", 'markersize':5, 'linestyle':':'}
kw_peigne = {'color': '#006400', 'label': 'Peigne Spatial  Théorie', 'linewidth': 2}
kw_peigne_exp = {'color': "#00AB00", 'label': 'Peigne spatial - Experience', 'marker':"o", 'markersize':5, 'linestyle':':'}
Title = "Spectre de la tension en sortie des micros\n(Fondamental à " + str(F) + " Hz)"

# --- Lecture des données experimentales ---
def lire_fichier(PATH):
    List_temps = []
    List_sig = []
    file_s2 = open(PATH)
    L_lignes = file_s2.readlines()
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


# --- Calcul des FFT ---
temps = np.linspace(-Te/2, Te/2, Ne)
V_s_th = mod.fem_singlecoil(Na, temps, A0, x0)
V_hp_th = mod.fem_humbucker(Na, N_hp, x_hp1, x_hp2, temps, A0, x0) / 2
V_hs_th = mod.fem_humbucker(Na, N_hs, x_hs1, x_hs2, temps, A0, x0) / 2

fft_s_th = mod.compute_and_normalize_fft(V_s_th, Ne)
fft_hp_th = mod.compute_and_normalize_fft(V_hp_th, Ne)
fft_hs_th = mod.compute_and_normalize_fft(V_hs_th, Ne)

# Nombre d'echantillon experimental
Ne_s = len(V_s_exp)
Ne_hp = len(V_hp_exp)
Ne_hs = len(V_hs_exp)

fft_s_exp_abs = mod.compute_and_normalize_fft(V_s_exp, Ne_s)
fft_hp_exp_abs = mod.compute_and_normalize_fft(V_hp_exp, Ne_hp)
fft_hs_exp_abs = mod.compute_and_normalize_fft(V_hs_exp, Ne_hs)

# Propager l'incertitude CSV vers la FFT (division par √N)
yerr_s_fft_abs = yerr_temp / np.sqrt(Ne_s)
yerr_hp_fft_abs = yerr_temp / np.sqrt(Ne_hp)
yerr_hs_fft_abs = yerr_temp / np.sqrt(Ne_hs)

# Tableau de fréquence pour chaque données
freqs_th = np.fft.fftfreq(Ne, 1 / (len(V_s_th) / (temps[-1] - temps[0])))
freqs_s_exp = np.fft.fftfreq(Ne_s, 1 / (len(V_s_exp) / (temps_s[-1] - temps_s[0])))
freqs_hp_exp = np.fft.fftfreq(Ne_hp, 1 / (len(V_hp_exp) / (temps_hp[-1] - temps_hp[0])))
freqs_hs_exp = np.fft.fftfreq(Ne_hs, 1 / (len(V_hs_exp) / (temps_hs[-1] - temps_hs[0])))

# Masque d'affichage entre 40Hz et 2000Hz
mask_th = (40 < freqs_th) & (freqs_th <= 2000)
mask_s = (40 < freqs_s_exp) & (freqs_s_exp <= 2000)
mask_hp = (40 < freqs_hp_exp) & (freqs_hp_exp <= 2000)
mask_hs = (40 < freqs_hs_exp) & (freqs_hs_exp <= 2000)

freqs_th = freqs_th[mask_th]
fft_s_th = mod.normalize_signal(fft_s_th[mask_th])
fft_hp_th = mod.normalize_signal(fft_hp_th[mask_th])
fft_hs_th = mod.normalize_signal(fft_hs_th[mask_th])

# Normaliser les FFT ET l'incertitude
max_fft_s = np.max(fft_s_exp_abs)
max_fft_hp = np.max(fft_hp_exp_abs)
max_fft_hs = np.max(fft_hs_exp_abs)

freqs_s_exp = freqs_s_exp[mask_s]
fft_s_exp_filtered = fft_s_exp_abs[mask_s]
fft_s_exp = fft_s_exp_filtered / max_fft_s 
yerr_s_exp = np.full_like(fft_s_exp_filtered, yerr_s_fft_abs) / max_fft_s

freqs_hp_exp = freqs_hp_exp[mask_hp]
fft_hp_exp_filtered = fft_hp_exp_abs[mask_hp]
fft_hp_exp = fft_hp_exp_filtered / max_fft_hp
yerr_hp_exp = np.full_like(fft_hp_exp_filtered, yerr_hp_fft_abs) / max_fft_hp

freqs_hs_exp = freqs_hs_exp[mask_hs]
fft_hs_exp_filtered = fft_hs_exp_abs[mask_hs]
fft_hs_exp = fft_hs_exp_filtered / max_fft_hs
yerr_hs_exp = np.full_like(fft_hs_exp_filtered, yerr_hs_fft_abs) / max_fft_hs

# --- Affichage des incertitudes moyennes sur les FFT ---
print("\n" + "="*60)
print("INCERTITUDES SUR LES FFT (DONNÉES NORMALISÉES)")
print("="*60)

print(f"SingleCoil:")
print(f"  - Incertitude moyenne: {np.mean(yerr_s_exp):.6e}")
print(f"  - Incertitude min/max: {np.min(yerr_s_exp):.6e} / {np.max(yerr_s_exp):.6e}")
print(f"  - Incertitude relative: {np.mean(yerr_s_exp / (fft_s_exp + 1e-12))*100:.4f}%")

print(f"\nHumbucker parallèle:")
print(f"  - Incertitude moyenne: {np.mean(yerr_hp_exp):.6e}")
print(f"  - Incertitude min/max: {np.min(yerr_hp_exp):.6e} / {np.max(yerr_hp_exp):.6e}")
print(f"  - Incertitude relative: {np.mean(yerr_hp_exp / (fft_hp_exp + 1e-12))*100:.4f}%")

print(f"\nHumbucker série:")
print(f"  - Incertitude moyenne: {np.mean(yerr_hs_exp):.6e}")
print(f"  - Incertitude min/max: {np.min(yerr_hs_exp):.6e} / {np.max(yerr_hs_exp):.6e}")
print(f"  - Incertitude relative: {np.mean(yerr_hs_exp / (fft_hs_exp + 1e-12))*100:.4f}%")

print("="*60 + "\n")

# --- Enregistrer les spectres ---
mod.save_plot(PATH_exp + "FFT/Theorie.png",
        (freqs_th, fft_s_th, kw_S_th),
        (freqs_th, fft_hp_th, kw_Hp_th),
        (freqs_th, fft_hs_th, kw_Hs_th),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Theorie.png"}")

mod.save_plot(PATH_exp + "FFT/Experience.png",
        (freqs_s_exp, fft_s_exp, {**kw_S_exp, 'yerr': yerr_s_exp}),
        (freqs_hp_exp, fft_hp_exp, {**kw_Hp_exp, 'yerr': yerr_hp_exp}),
        (freqs_hs_exp, fft_hs_exp, {**kw_Hs_exp, 'yerr': yerr_hs_exp}),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Experience.png"}")

mod.save_plot(PATH_exp + "FFT/Singlecoil.png",
        (freqs_s_exp, fft_s_exp, {**kw_S_exp, 'yerr': yerr_s_exp}),
        (freqs_th, fft_s_th, kw_S_th),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Singlecoil.png"}")

mod.save_plot(PATH_exp + "FFT/Humbucker_parra.png",
        (freqs_hp_exp, fft_hp_exp, {**kw_Hp_exp, 'yerr': yerr_hp_exp}),
        (freqs_th, fft_hp_th, kw_Hp_th),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Humbucker.png"}")

mod.save_plot(PATH_exp + "FFT/Humbucker_serie.png",
        (freqs_hs_exp, fft_hs_exp, {**kw_Hs_exp, 'yerr': yerr_hs_exp}),
        (freqs_th, fft_hs_th, kw_Hs_th),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee", error_bars=True)
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Humbucker.png"}")

# --- Mise en evidence du filtrage spatial ---
freqs = np.linspace(10, 20000, 1000)
L_freqs, L_pic_s_exp = mod.extraire_pics_separes(freqs_s_exp, fft_s_exp, prominence=0.04, distance=5)
L_freqs, L_pic_h_exp = mod.extraire_pics_separes(freqs_hs_exp, fft_hs_exp, prominence=0.03, distance=5)
peigne_exp = 20*np.log10(L_pic_h_exp / L_pic_s_exp)
ratio_th_hs, peigne_hs = mod.attenuation_theorique_db(freqs, x_hs1, x_hs2)

mod.save_plot(PATH_exp + "Filtrage_spatial.png",
    (freqs, ratio_th_hs, {'color': "#444444", 'label': "Gain Theorique S [$dB$]", "linewidth": 0.5}),
    (L_freqs, peigne_exp, kw_peigne_exp),
    (freqs, peigne_hs, kw_peigne),
    title=f"Influence de la position du micro sur le spectre\nFiltrage spatial pour un fondamental à {F}$Hz$", 
    xlabel="Frequence [$Hz$]", ylabel="Gain relatif [$dB$]", xlim=(0, 20_000), log=True, vspan=True)