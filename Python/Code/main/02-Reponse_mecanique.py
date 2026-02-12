"""Modele de la vibration d'une corde et de la tension induite dans un micro simple
et un humbucker.
Auteur : Christophe Morel
Date : Decembre 2025
"""


print("\n\n---\nModele_fem.py\n---\n")
print("-" * 100)


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
from scipy.fft import fft, fftfreq
plt.close("all")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
print("Bibliotheques importees.")

                                            #!####################!#
                                            #!    Constantes      !#
                                            #!####################!#

print("-" * 100)
print("\nInitialisation des constantes")
#* Constante pour code
tracer_fft = True
PATH_exp = "Python/Exports/02-Reponse_mecanique/"
#* Constante physique
rho = 1.68 * 10**(-8)   # Masse volumique du cuivre (Ohm.m)
mu0 = 4e-7 * np.pi     # Permeabilite magnetique du vide (H.m-1)
mu_r = 3.5
K=0.7
#* Parametre maquette
xm = 0.11              # Position du micro singlecoil
x1 = 0.147              # Position du micro humbucker 1
x2 = 0.165              # Position du micro humbucker 2
L = 0.845               # Longueur de la corde
mu = 9 * 10**(-3)       # Masse lineique de la corde
dBdy = 0.03             #? Gradient du champ magnetique (A verifier ???)
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
#* Parametre variable
A = 3e-3          #? Amplitude initiale de vibration
x0 = 0.140                 #? Origine de la perturbation
F = 78.0                   #? Frequence de la note jouee
T = 4 * mu * (L * F)**2    #? Tension de la corde
#* Couleurs graphiques
kw_Hp_th = {'color': '#000080', 'label': 'Humbucker parallèle - Théorie', 'linewidth': 2}
kw_S_th = {'color': '#8B0000', 'label': 'SingleCoil - Théorie', 'linewidth': 2}
kw_Hs_th = {'color': '#006400', 'label': 'Humbucker série - Théorie', 'linewidth': 2}
kw_Hp_exp = {'color': '#9400D3', 'label': 'Humbucker parallèle - experience', 'marker':"o", 'markersize':5, 'linestyle':':'}
kw_Hs_exp = {'color': '#00CD00', 'label': 'Humbucker série - experience', 'marker':"o", 'markersize':5, 'linestyle':':'}
kw_S_exp = {'color': '#FF8C00', 'label': 'SingleCoil - experience', 'marker':"o", 'markersize':5, 'linestyle':':'}

if RCC_h is None or l_h is None or N_h is None:
    logging.warning("Parametres humbucker (RCC_h, l_h, N_h) non defini(s). Certaines simulations pourront utiliser N_s en substitution.")

print("Constantes initialisees.")
print("\nValeurs des parametres (avec unites) :")

params = [
    ("rho", rho, "Ohm.m", "Resistivite du fil de cuivre"),
    ("mu0", mu0, "H.m-1", "Permeabilite magnetique du vide"),
    ("xm", xm, "m", "Distance corde-micro"),
    ("x1", x1, "m", "Distance corde-micro humbucker 1"),
    ("x2", x2, "m", "Distance corde-micro humbucker 2"),
    ("L", L, "m", "Longueur de la corde"),
    ("mu", mu, "kg.m-1", "Masse lineique de la corde"),
    ("dBdy", dBdy, "T.m-1", "Gradient du champ magnetique"),
    ("S", S, "m2", "Surface de la bobine"),
    ("RCC_s", RCC_s, "Ohm", "Resistance DC micro singlecoil"),
    ("l_s", l_s, "m", "Longueur moyenne du fil enroule"),
    ("N_s", N_s, "spires", "Nombre de spires du micro singlecoil"),
    ("RCC_h", RCC_h, "Ohm", "Resistance DC micro humbucker"),
    ("l_h", l_h, "m", "Longueur moyenne du fil humbucker"),
    ("N_h", N_h, "spires", "Nombre de spires micro humbucker"),
    ("A", A, "m", "Amplitude initiale de vibration"),
    ("x0", x0, "m", "Origine de la perturbation"),
    ("F", F, "Hz", "Frequence de la note jouee"),
    ("T", T, "N", "Tension de la corde"),
]

try :
    print(f"{'Var':<8} {'Valeur':>18} {'Unite':<10} {'Description':<45}")
    print("-" * 85)
    compteur = 1
    for var, val, unit, desc in params:
        if isinstance(val, (int, np.integer)):
            val_str = f"{val:d}"
        elif isinstance(val, (float, np.floating)):
            val_str = f"{val:.6g}"
        else:
            val_str = str(val)
        if compteur in (1, 3, 9, 11, 14, 17):
            print("-" * 70)
        print(f"{var:<8} {val_str:>18} {unit:<10} {desc:<45}")
        compteur += 1
except Exception as e:
    logging.error(f"Erreur lors de l'affichage des parametres : {e}")

print("-" * 85)


                                            #!####################!#
                                            #!      Get data      !#
                                            #!####################!#

path = "Python/data/raw/SingleCoil_corde_78,5.csv"
try:
    with open(path) as doc:
        L_lignes = doc.readlines()
    Test_s_fft = []
    Test_temps_s_fft = []
    for i in range(35200, 60200):
        values = L_lignes[i].strip().split(";")
        values = [v.replace(',', '.') for v in values]
        values = [float(v.replace("\n", "")) for v in values]
        Test_temps_s_fft.append(values[0])
        Test_s_fft.append(values[1])
    Test_temps_s_fft = np.array(Test_temps_s_fft)
    Test_temps_s_fft -= Test_temps_s_fft[0]
    Test_s_fft = np.array(Test_s_fft)
    print("Donnees de test Single Coil corde seule lues.")
except Exception as e:
    logging.error(f"Erreur lors de la lecture des donnees de test Single Coil corde seule : {e}")

path = "Python/data/raw/Humbucker_corde_78,5.csv"
try:
    with open(path) as doc:
        L_lignes = doc.readlines()
    Test_h_fft = []
    Test_temps_h_fft = []
    for i in range(58935, 83935):
        values = L_lignes[i].strip().split(";")
        values = [v.replace(',', '.') for v in values]
        values = [float(v.replace("\n", "")) for v in values]
        Test_temps_h_fft.append(values[0])
        Test_h_fft.append(values[1])
    Test_temps_h_fft = np.array(Test_temps_h_fft)
    Test_temps_h_fft -= Test_temps_h_fft[0]
    Test_h_fft = np.array(Test_h_fft)
    print("Donnees de test Humbucker corde seule lues.")
except Exception as e:
    logging.error(f"Erreur lors de la lecture des donnees de test Humbucker corde seule : {e}")



print("\nCalcul des tensions induites...")
print("\nEchantillonnage...")
Na = 20
Te = 4000 * 0.00001
Ne = 4000
fs = Ne / Te
Te_fft = 1
Ne_fft = 25_000
fs_fft = Ne_fft / Te_fft
# Affichage des parametres d'echantillonnage
print("\nParametres d'echantillonnage (avec unites) :")
print(f"Te = {Te} s")
print(f"Ne = {Ne} echantillons")
print(f"fs = {fs:.6g} Hz")
print(f"Na = {Na}")
print("\nCalcul des tensions en reponse a la vibration de la corde...")

temps = np.linspace(0, Te, Ne, endpoint=False)
fem_s = mod.fem_singlecoil(Na, N_s, xm, temps, A, L, T, mu, x0, S, dBdy)
fem_h = mod.fem_humbucker(Na, N_h, x1, x2, temps, A, L, T, mu, x0, S, dBdy)



                                            #!####################!#
                                            #!       Exports      !#
                                            #!####################!#


# Ensure export directory exists
os.makedirs("Python/Exports", exist_ok=True)
print("-" * 100)


if tracer_fft:
    print("-" * 100)
    print("\nCalcul des FFT...")
    print("\nParametres d'echantillonnage (avec unites) :")
    print(f"Te = {Te_fft} s")
    print(f"Ne = {Ne_fft} echantillons")
    print(f"fs = {fs_fft:.6g} Hz")
    print(f"Na = {Na}")
    print("\nCalcul des tensions en reponse a la vibration de la corde...")
    
    temps_fft = Test_temps_h_fft
    fem_s_fft = mod.fem_singlecoil(Na, N_s, xm, temps_fft, A, L, T, mu, x0, S, dBdy)
    fem_h_fft = mod.fem_humbucker(Na, N_h, x1, x2, temps_fft, A, L, T, mu, x0, S, dBdy)
    
    fft_s_fem = mod.compute_and_normalize_fft(fem_s_fft, Ne_fft)
    fft_h_fem = mod.compute_and_normalize_fft(fem_h_fft, Ne_fft)
    fft_h_test = mod.compute_and_normalize_fft(Test_h_fft, len(Test_h_fft))
    fft_s_test = mod.compute_and_normalize_fft(Test_s_fft, len(Test_s_fft))

    
    freqs = np.fft.fftfreq(len(Test_h_fft), 1 / (len(Test_h_fft) / (Test_temps_h_fft[-1] - Test_temps_h_fft[0])))
    mask = 0 <= freqs
    freqs = freqs[mask]
    fft_s_fem = fft_s_fem[mask]
    fft_h_fem = fft_h_fem[mask]
    fft_h_test = fft_h_test[mask]
    fft_s_test = fft_s_test[mask]
    
    mask2 = freqs <= 800
    freqs = freqs[mask2]
    fft_s_fem = fft_s_fem[mask2]
    fft_h_fem = fft_h_fem[mask2]
    fft_h_test = fft_h_test[mask2]
    fft_s_test = fft_s_test[mask2]
    
    
    fft_s_fem = fft_s_fem / np.max(fft_s_fem)
    fft_h_fem = fft_h_fem / np.max(fft_h_fem)
    fft_h_test = fft_h_test / np.max(fft_h_test)
    fft_s_test = fft_s_test / np.max(fft_s_test)

    print("FFT calculees.")
    Title = "Spectre de la tension en sortie des micros\n(Fondamental à " + str(F) + " Hz)"
    print("Affichage des FFT...")
    
    #-- 01 --#
    mod.save_plot(PATH_exp + "FFT/Theorie.png",
            (freqs, fft_h_fem, kw_Hp_th),
            (freqs, fft_s_fem, kw_S_th),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
    print(f"FFT sauvegardees sous {PATH_exp + "FFT/Theorie.png"}")
    
    mod.save_plot(PATH_exp + "FFT/Experience.png",
            (freqs, fft_s_test, kw_S_exp),
            (freqs, fft_h_test, kw_Hp_exp),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
    print(f"FFT sauvegardees sous {PATH_exp + "FFT/Experience.png"}")

    mod.save_plot(PATH_exp + "FFT/Singlecoil.png",
            (freqs, fft_s_test, kw_S_exp),
            (freqs, fft_s_fem, kw_S_th),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
    print(f"FFT sauvegardees sous {PATH_exp + "FFT/Singlecoil.png"}")
    
    mod.save_plot(PATH_exp + "FFT/Humbucker.png",
            (freqs, fft_h_test, kw_Hp_exp),
            (freqs, fft_h_fem, kw_Hp_th),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
    print(f"FFT sauvegardees sous {PATH_exp + "FFT/Humbucker.png"}")

print("-" * 100)
print("\nAffichage des courbes...")
#-- 01 --#

#* --- Attenuation --- *#
L_freqs, L_pic_s_exp = mod.extraire_pics_separes(freqs, fft_s_test)
L_freqs, L_pic_h_exp = mod.extraire_pics_separes(freqs, fft_h_test)

freqs = np.linspace(20, 20_000, 1000)
ratio_exp = 20*np.log10(L_pic_h_exp / L_pic_s_exp)
ratio_th, peigne = mod.attenuation_theorique_db(freqs, x1, x2, xm, L, F)
"""
mod.save_plot(PATH_exp + "Filtrage_spatial.png",
    (freqs, ratio_th, {'color': "#444444", 'label': "Gain Theorique [$dB$]", "linewidth": 1.5}),
    (L_freqs, ratio_exp, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': '', 'color': Color_h_test, 'label': "Gain experimentale [$dB$]"}),
    (freqs, peigne, {'color': Color_h, 'label': "Filtrage en peigne spatial [$dB$]", "linewidth": 2}),
    title=f"Influence de la position du micro sur le spectre\nFiltrage spatial pour un fondamental à {F}$Hz$", xlabel="Frequence [$Hz$]", ylabel="Gain relatif [$dB$]", log=True, vspan=True)
"""
#* --- Comparaison des micros Theorique (total) ---*#
mod.save_plot(PATH_exp + "temporel/Theorique.png",
        (temps, fem_h, kw_Hp_th),
        (temps, fem_s, kw_S_th),
        title="Tension induite dans les micros\n(Fondamental à " + str(F) + " Hz)",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print(f"Courbes sauvegardees sous {PATH_exp + "temporel/Theorique.png"}")

#* --- Micro Singlecoil (corde) ---*#
mod.save_plot(PATH_exp + "temporel/Singlecoil.png",
        (Test_temps_s_fft[:4000], Test_s_fft[:4000], kw_S_exp),
        (temps, -fem_s, kw_S_th),
        title="Tension induite dans le micro Singlecoil\n(Fondamental à " + str(F) + " Hz)",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print(f"Courbes sauvegardees sous {PATH_exp + "temporel/Singlecoil.png"}")

#* --- Micro Humbucker (corde) ---*#
mod.save_plot(PATH_exp + "temporel/Humbucker.png",
        (Test_temps_h_fft[:4000], Test_h_fft[:4000], kw_Hp_exp),
        (temps, -fem_h, kw_Hp_th),
        title="Tension induite dans le micro Humbucker\n(Fondamental à " + str(F) + " Hz)",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print(f"Courbes sauvegardees sous {PATH_exp + "temporel/Humbucker.png"}")
