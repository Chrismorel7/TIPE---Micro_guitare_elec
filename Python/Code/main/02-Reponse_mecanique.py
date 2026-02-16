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

# --- Constante pour code ---
tracer_fft = True
PATH_exp = "Python/Exports/02-Reponse_mecanique/"
Na = 20
Te = 1
Ne = 10000
fs = Ne / Te

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
kw_Hp_exp = {'color': '#9400D3', 'label': 'Humbucker parallèle - experience', 'marker':"o", 'markersize':5, 'linestyle':':'}
kw_Hs_exp = {'color': "#00AB00", 'label': 'Humbucker série - experience', 'marker':"o", 'markersize':5, 'linestyle':':'}
kw_S_exp = {'color': '#FF8C00', 'label': 'SingleCoil - experience', 'marker':"o", 'markersize':5, 'linestyle':':'}
kw_peigne = {'color': '#006400', 'label': 'Peigne Spatial  Théorie', 'linewidth': 2}
kw_peigne_exp = {'color': "#00AB00", 'label': 'Peigne spatial - Experience', 'marker':"o", 'markersize':5, 'linestyle':':'}

                                            #!####################!#
                                            #!      Get data      !#
                                            #!####################!#

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

print("\nCalcul des tensions induites...")
print("\nEchantillonnage...")
print("\nParametres d'echantillonnage (avec unites) :")
print(f"Te = {Te} s")
print(f"Ne = {Ne} echantillons")
print(f"fs = {fs:.6g} Hz")
print(f"Na = {Na} mode consideres")
print("\nCalcul des tensions en reponse a la vibration de la corde...")

temps = np.linspace(-Te/2, Te/2, Ne)
V_s_th = mod.fem_singlecoil(Na, temps, A0, x0)
V_hp_th = mod.fem_humbucker(Na, N_hp, x_hp1, x_hp2, temps, A0, x0) / 2
V_hs_th = mod.fem_humbucker(Na, N_hs, x_hs1, x_hs2, temps, A0, x0) / 2


                                            #!####################!#
                                            #!       Exports      !#
                                            #!####################!#


# Verifier si le fichier d'exportation existe
os.makedirs("Python/Exports", exist_ok=True)

fft_s_th = mod.compute_and_normalize_fft(V_s_th, Ne)
fft_hp_th = mod.compute_and_normalize_fft(V_hp_th, Ne)
fft_hs_th = mod.compute_and_normalize_fft(V_hs_th, Ne)

# Recalculate with actual experimental data sizes
Ne_s = len(V_s_exp)
Ne_hp = len(V_hp_exp)
Ne_hs = len(V_hs_exp)

fft_s_exp = mod.compute_and_normalize_fft(V_s_exp, Ne_s)
fft_hp_exp = mod.compute_and_normalize_fft(V_hp_exp, Ne_hp)
fft_hs_exp = mod.compute_and_normalize_fft(V_hs_exp, Ne_hs)

# Create frequency arrays for each dataset
freqs_th = np.fft.fftfreq(Ne, 1 / (len(V_s_th) / (temps[-1] - temps[0])))
freqs_s_exp = np.fft.fftfreq(Ne_s, 1 / (len(V_s_exp) / (temps_s[-1] - temps_s[0])))
freqs_hp_exp = np.fft.fftfreq(Ne_hp, 1 / (len(V_hp_exp) / (temps_hp[-1] - temps_hp[0])))
freqs_hs_exp = np.fft.fftfreq(Ne_hs, 1 / (len(V_hs_exp) / (temps_hs[-1] - temps_hs[0])))

# Apply masks separately for theory and each experiment
mask_th = (40 < freqs_th) & (freqs_th <= 800)
mask_s = (40 < freqs_s_exp) & (freqs_s_exp <= 800)
mask_hp = (40 < freqs_hp_exp) & (freqs_hp_exp <= 800)
mask_hs = (40 < freqs_hs_exp) & (freqs_hs_exp <= 800)

# Filter all arrays consistently
freqs_th = freqs_th[mask_th]
fft_s_th = mod.normalize_signal(fft_s_th[mask_th])
fft_hp_th = mod.normalize_signal(fft_hp_th[mask_th])
fft_hs_th = mod.normalize_signal(fft_hs_th[mask_th])

freqs_s_exp = freqs_s_exp[mask_s]
fft_s_exp = mod.normalize_signal(fft_s_exp[mask_s])

freqs_hp_exp = freqs_hp_exp[mask_hp]
fft_hp_exp = mod.normalize_signal(fft_hp_exp[mask_hp])

freqs_hs_exp = freqs_hs_exp[mask_hs]
fft_hs_exp = mod.normalize_signal(fft_hs_exp[mask_hs])

print("FFT calculees.")
Title = "Spectre de la tension en sortie des micros\n(Fondamental à " + str(F) + " Hz)"
print("Affichage des FFT...")

mod.save_plot(PATH_exp + "FFT/Theorie.png",
        (freqs_th, fft_s_th, kw_S_th),
        (freqs_th, fft_hp_th, kw_Hp_th),
        (freqs_th, fft_hs_th, kw_Hs_th),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Theorie.png"}")

mod.save_plot(PATH_exp + "FFT/Experience.png",
        (freqs_s_exp, fft_s_exp, kw_S_exp),
        (freqs_hp_exp, fft_hp_exp, kw_Hp_exp),
        (freqs_hs_exp, fft_hs_exp, kw_Hs_exp),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Experience.png"}")
mod.save_plot(PATH_exp + "FFT/Singlecoil.png",
        (freqs_s_exp, fft_s_exp, kw_S_exp),
        (freqs_th, fft_s_th, kw_S_th),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Singlecoil.png"}")

mod.save_plot(PATH_exp + "FFT/Humbucker_parra.png",
        (freqs_hp_exp, fft_hp_exp, kw_Hp_exp),
        (freqs_th, fft_hp_th, kw_Hp_th),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Humbucker.png"}")

mod.save_plot(PATH_exp + "FFT/Humbucker_serie.png",
        (freqs_hs_exp, fft_hs_exp, kw_Hs_exp),
        (freqs_th, fft_hs_th, kw_Hs_th),
        title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
print(f"FFT sauvegardees sous {PATH_exp + "FFT/Humbucker.png"}")

print("\nAffichage des courbes...")

#* --- Attenuation --- *#
#* --- Attenuation --- *#

# Réduire prominence et distance pour détecter plus de pics
freqs = np.linspace(10, 20000, 1000)
L_freqs, L_pic_s_exp = mod.extraire_pics_separes(freqs_s_exp, fft_s_exp, prominence=0.02, distance=5)
L_freqs, L_pic_h_exp = mod.extraire_pics_separes(freqs_hs_exp, fft_hs_exp, prominence=0.02, distance=5)

peigne_exp = 20*np.log10(L_pic_h_exp / L_pic_s_exp)
ratio_th, peigne = mod.attenuation_theorique_db(freqs, x_hs1, x_hs2)

mod.save_plot(PATH_exp + "Filtrage_spatial.png",
    (freqs, ratio_th, {'color': "#444444", 'label': "Gain Theorique [$dB$]", "linewidth": 0.5}),
    (L_freqs, peigne_exp, kw_peigne_exp),
    (freqs, peigne, kw_peigne),
    title=f"Influence de la position du micro sur le spectre\nFiltrage spatial pour un fondamental à {F}$Hz$", 
    xlabel="Frequence [$Hz$]", ylabel="Gain relatif [$dB$]", xlim=(0, 20_000), log=True, vspan=True)

Te = 0.1
Ne = 1000
fs = Ne / Te

T_fundamental = 1 / F  # Période fondamentale
t_range = 3 * T_fundamental  # 3 périodes
t_min = -t_range / 2
t_max = t_range / 2

print(f"Période fondamentale: {T_fundamental:.6f} s")
print(f"Plage affichée: [{t_min:.6f}, {t_max:.6f}] s")
print(f"Durée affichée: {t_range:.6f} s (3 périodes)")

# Générer les temps théoriques avec la bonne résolution
temps = np.linspace(-Te/2, Te/2, Ne)
V_s_th = mod.fem_singlecoil(Na, temps, A0, x0)
V_hp_th = mod.fem_humbucker(Na, N_hp, x_hp1, x_hp2, temps, A0, x0) / 2
V_hs_th = mod.fem_humbucker(Na, N_hs, x_hs1, x_hs2, temps, A0, x0) / 2

# Créer les masques pour les plages temporelles
mask_temps_th = (temps >= t_min) & (temps <= t_max)
mask_temps_s = (temps_s >= t_min) & (temps_s <= t_max)
mask_temps_hp = (temps_hp >= t_min) & (temps_hp <= t_max)
mask_temps_hs = (temps_hs >= t_min) & (temps_hs <= t_max)

# Filtrer les données
temps_th = temps[mask_temps_th]
V_s_th = V_s_th[mask_temps_th]
V_hp_th = V_hp_th[mask_temps_th]
V_hs_th = V_hs_th[mask_temps_th]

temps_s = temps_s[mask_temps_s]
V_s_exp = V_s_exp[mask_temps_s]

temps_hp = temps_hp[mask_temps_hp]
V_hp_exp = V_hp_exp[mask_temps_hp]

temps_hs = temps_hs[mask_temps_hs]
V_hs_exp = V_hs_exp[mask_temps_hs]

# --- Comparaison des micros Théorie ---
mod.save_plot(PATH_exp + "temporel/Theorique.png",
        (temps_th, V_s_th, kw_S_th),
        (temps_th, V_hp_th, kw_Hp_th),
        (temps_th, V_hs_th, kw_Hs_th),
        title="Tension induite dans les micros - 3 périodes\n(Fondamental à " + str(F) + " Hz)",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print(f"Courbes sauvegardées sous {PATH_exp}temporel/Theorique.png")

# --- Micro Singlecoil ---
mod.save_plot(PATH_exp + "temporel/Singlecoil.png",
        (temps_s, V_s_exp, kw_S_exp),
        (temps_th, V_s_th, kw_S_th),
        title="Tension induite dans le micro SingleCoil - 3 périodes\n(Fondamental à " + str(F) + " Hz)",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print(f"Courbes sauvegardées sous {PATH_exp}temporel/Singlecoil.png")

# --- Micro Humbucker parallèle ---
mod.save_plot(PATH_exp + "temporel/Humbucker_parra.png",
        (temps_hp, V_hp_exp, kw_Hp_exp),
        (temps_th, V_hp_th, kw_Hp_th),
        title="Tension induite dans le micro Humbucker parallèle - 3 périodes\n(Fondamental à " + str(F) + " Hz)",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print(f"Courbes sauvegardées sous {PATH_exp}temporel/Humbucker_parra.png")

# --- Micro Humbucker série ---
mod.save_plot(PATH_exp + "temporel/Humbucker_serie.png",
        (temps_hs, V_hs_exp, kw_Hs_exp),
        (temps_th, V_hs_th, kw_Hs_th),
        title="Tension induite dans le micro Humbucker série - 3 périodes\n(Fondamental à " + str(F) + " Hz)",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print(f"Courbes sauvegardées sous {PATH_exp}temporel/Humbucker_serie.png")

print("\nAffichage temporel terminé !")