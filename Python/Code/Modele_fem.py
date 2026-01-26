"""Modele de la vibration d'une corde et de la tension induite dans un micro simple
et un humbucker.
Auteur : Christophe Morel
Date : Decembre 2025
"""

print("\n\n---\nModele_fem.py\n---\n")

#! ------------------ Bibliotheques ------------------------------
print("-" * 100)
print("\nImportation des bibliotheques...")
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
A = 70 * 10**(-4)          #? Amplitude initiale de vibration
x0 = 0.140                 #? Origine de la perturbation
F = 78.0                   #? Frequence de la note jouee
T = 4 * mu * (L * F)**2    #? Tension de la corde
#* Couleurs graphiques
Color_c = "#505050"
Color_h_test = "#008B8B"
Color_s_test = "#FF8C00"
ls_exp='--'
lw_exp=3
m_exp="+"
ms_exp=10
Color_h = "#000080"
Color_s = "#8B0000"
ls_th='-'
lw_th=2

if RCC_h is None or lmoy_h is None or N_h is None:
    logging.warning("Parametres humbucker (RCC_h, lmoy_h, N_h) non defini(s). Certaines simulations pourront utiliser N_s en substitution.")

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
    ("d", d, "m", "Diametre du fil de cuivre"),
    ("S", S, "m2", "Surface de la bobine"),
    ("RCC_s", RCC_s, "Ohm", "Resistance DC micro singlecoil"),
    ("lmoy_s", lmoy_s, "m", "Longueur moyenne du fil enroule"),
    ("N_s", N_s, "spires", "Nombre de spires du micro singlecoil"),
    ("RCC_h", RCC_h, "Ohm", "Resistance DC micro humbucker"),
    ("lmoy_h", lmoy_h, "m", "Longueur moyenne du fil humbucker"),
    ("N_h", N_h, "spires", "Nombre de spires micro humbucker"),
    ("Nbob", Nbob, "spires", "Nombre de spires bobine Helmholtz"),
    ("R", R, "m", "Rayon des bobines Helmholtz"),
    ("Lbob", Lbob, "H", "Inductance bobines Helmholtz"),
    ("Rbob", Rbob, "Ohm", "Resistance bobines Helmholtz"),
    ("fB", fB, "Hz", "Frequence du champ parasite"),
    ("U0", U0, "V", "Tension bobines Helmholtz"),
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
        if compteur in (1, 3, 9, 11, 14, 17, 24):
            print("-" * 70)
        print(f"{var:<8} {val_str:>18} {unit:<10} {desc:<45}")
        compteur += 1
except Exception as e:
    logging.error(f"Erreur lors de l'affichage des parametres : {e}")

print("-" * 85)

# ------------------ Utilitaires (lecture / plot / fft) ----------
def read_test_file(path: str, line_start: int, line_end: int):
    """Lit les lignes [line_start, line_end) d'un fichier tabule et renvoie deux tableaux temps,valeur."""
    with open(path, 'r') as doc:
        L_lignes = doc.readlines()
    times = []
    vals = []
    for i in range(line_start, line_end):
        values = L_lignes[i].strip().split("\t")
        values = [float(v.replace("\n", "")) for v in values]
        times.append(values[0])
        vals.append(values[1])
    times = np.array(times)
    times -= times[0]
    return times, np.array(vals)


def read_test_file_partial(path: str, line_start: int, line_end: int, keep_all=False):
    """Version qui lit et n'enlève pas la ligne 0 si besoin (utilise pour petits fichiers)."""
    with open(path, 'r') as doc:
        L_lignes = doc.readlines()
    times = []
    vals = []
    for i in range(line_start, line_end):
        values = L_lignes[i].strip().split("\t")
        values = [float(v.replace("\n", "")) for v in values]
        times.append(values[0])
        vals.append(values[1])
    times = np.array(times)
    times -= times[0]
    return times, np.array(vals)


# ------------------ Resultats / Execution ----------------------

print("Lecture des donnees de test...")
List_Bpar = np.array([0,5.5, 11, 16.5, 22, 27.5, 33, 38.5, 44, 49.5]) * 10**-6
List_amp_h_th = np.array([N_h*S*2*np.pi*fB*B*alpha*abs(x1-x2) for B in List_Bpar])
List_amp_s_th = np.array([N_s*S*2*np.pi*fB*B for B in List_Bpar])

List_amp_h_test = [0]
List_amp_s_test = [0]

file = open("Python/Code/amplitudes.csv")
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

[k_h_exp, v0h] = np.polyfit(List_Bpar, List_amp_h_test, 1)
[k_s_exp, v0s] = np.polyfit(List_Bpar, List_amp_s_test, 1)
List_amp_h_test_lin = k_h_exp * List_Bpar
List_amp_s_test_lin = k_s_exp * List_Bpar

ks = N_s*S*2*np.pi*fB
kh = N_h*S*2*np.pi*fB*alpha*abs(x1-x2)

doc_path = "Python/filtered data/SingleCoil_par_17,5V.csv"
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

path = "Python/raw data/SingleCoil_corde_78,5.csv"
try:
    with open(path) as doc:
        L_lignes = doc.readlines()
    Test_temps_s = []
    Test_s = []
    for i in range(35200, 60200):
        values = L_lignes[i].strip().split(";")
        values = [v.replace(',', '.') for v in values]
        values = [float(v.replace("\n", "")) for v in values]
        Test_temps_s.append(values[0])
        Test_s.append(values[1])
    Test_temps_s = np.array(Test_temps_s)
    Test_temps_s -= Test_temps_s[0]
    Test_temps_s_fft = np.array(Test_temps_s)
    Test_temps_s = np.array(Test_temps_s[:4000])
    Test_s_fft = np.array(Test_s)
    Test_s = np.array(Test_s[:4000])
    print("Donnees de test Single Coil corde seule lues.")
except Exception as e:
    logging.error(f"Erreur lors de la lecture des donnees de test Single Coil corde seule : {e}")
try:
    Test_s_tot = np.array(Test_s_p[:4000]) + np.array(Test_s[:4000])
    Test_temps_s_tot = np.array(Test_temps_s[:4000])
    
    Test_s_tot_fft = np.array(Test_s_p) + np.array(Test_s_fft)
    Test_temps_s_tot_fft = np.array(Test_temps_s_fft)
    print("Donnees de test Single Coil parasite + corde lues.")
except Exception as e:
    logging.error(f"Erreur lors de la lecture des donnees de test Single Coil parasite + corde : {e}")

path = "Python/raw data/Humbucker_corde_78,5.csv"
try:
    with open(path) as doc:
        L_lignes = doc.readlines()
    Test_temps_h = []
    Test_h = []
    for i in range(58935, 83935):
        values = L_lignes[i].strip().split(";")
        values = [v.replace(',', '.') for v in values]
        values = [float(v.replace("\n", "")) for v in values]
        Test_temps_h.append(values[0])
        Test_h.append(values[1])
    Test_temps_h = np.array(Test_temps_h)
    Test_temps_h -= Test_temps_h[0]
    Test_temps_h_fft = np.array(Test_temps_h)
    Test_temps_h = np.array(Test_temps_h[:4000])
    Test_h_fft = np.array(Test_h)
    Test_h = - np.array(Test_h[:4000])
    print("Donnees de test Humbucker corde seule lues.")
except Exception as e:
    logging.error(f"Erreur lors de la lecture des donnees de test Humbucker corde seule : {e}")

doc_path = "Python/filtered data/Humbucker_par_8V.csv"
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

try:
    Test_h_tot = np.array(Test_h_p[:4000]) + np.array(Test_h[:4000])
    Test_temps_h_tot = np.array(Test_temps_h[:4000])
    
    Test_h_tot_fft = np.array(Test_h_p) + np.array(Test_h_fft)
    Test_temps_h_tot_fft = np.array(Test_temps_h_fft)
    print("Donnees de test Humbucker parasite + corde lues.")
except Exception as e:
    logging.error(f"Erreur lors de la lecture des donnees de test Humbucker parasite + corde : {e}")
print("-" * 100)


# Ensure export directory exists
os.makedirs("Python/Exports", exist_ok=True)
print("-" * 100)


print("\nCalcul des tensions induites...")
print("\nEchantillonnage...")
Na = 20
Te = 4000 * 0.00001
Ne = 4000
fs = Ne / Te
Te_fft = 1
Ne_fft = 100_000
fs_fft = Ne_fft / Te_fft
# Affichage des parametres d'echantillonnage
print("\nParametres d'echantillonnage (avec unites) :")
print(f"Te = {Te} s")
print(f"Ne = {Ne} echantillons")
print(f"fs = {fs:.6g} Hz")
print(f"Na = {Na}")
print("\nCalcul des tensions en reponse a la vibration de la corde...")


temps = np.linspace(0, Te, Ne, endpoint=False)
corde = mod.entree(Na, xm, temps, A, L, T, mu, x0)
fem_s = mod.fem_singlecoil(Na, N_s, xm, temps, A, L, T, mu, x0, S, dBdy)
fem_h = mod.fem_humbucker(Na, N_h, x1, x2, temps, A, L, T, mu, x0, S, dBdy)


print("\nCalcul des tensions en reponse au champ parasite...")
fem_s_par = mod.fem_par_s(N_s, fB, temps, B0, S, 2*np.pi/3)
fem_h_par = mod.fem_par_h(N_h, x1, x2, fB, temps, B0, alpha, S)


print("\nCalcul des tensions totales...")
fem_s_tot = fem_s + fem_s_par
fem_h_tot = fem_h + fem_h_par
tracer_fft = True
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
    corde_fft = mod.entree(Na, xm, Test_temps_h_fft, A, L, T, mu, x0)
    fem_s_fft = mod.fem_singlecoil(Na, N_s, xm, Test_temps_h_fft, A, L, T, mu, x0, S, dBdy)
    fem_h_fft = mod.fem_humbucker(Na, N_h, x1, x2, Test_temps_h_fft, A, L, T, mu, x0, S, dBdy)
    print("\nCalcul des tensions en reponse au champ parasite...")
    fem_s_par_fft = mod.fem_par_s(N_s, fB, temps_fft, B0, S, 2*np.pi/3)
    fem_h_par_fft = mod.fem_par_h(N_h, x1, x2, fB, temps_fft, B0, alpha, S)
    print("\nCalcul des tensions totales...")
    fem_s_tot_fft = fem_s_fft + fem_s_par_fft
    fem_h_tot_fft = fem_h_fft + fem_h_par_fft
    fft_s_amp_fem = mod.compute_and_normalize_fft(fem_s_fft, Ne_fft)
    fft_h_amp_fem = mod.compute_and_normalize_fft(fem_h_fft, Ne_fft)
    fft_h_test = mod.compute_and_normalize_fft(Test_h_fft, Ne_fft)
    fft_s_test = mod.compute_and_normalize_fft(Test_s_fft, Ne_fft)
    fft_s_amp_tot = mod.compute_and_normalize_fft(fem_s_tot_fft, Ne_fft)
    fft_h_amp_tot = mod.compute_and_normalize_fft(fem_h_tot_fft, Ne_fft)
    fft_h_test_tot = mod.compute_and_normalize_fft(Test_h_tot_fft, Ne_fft)
    fft_s_test_tot = mod.compute_and_normalize_fft(Test_s_tot_fft, Ne_fft)

    
    freqs = np.fft.fftfreq(25000, 1 / fs_fft)
    mask = 0 <= freqs
    freqs = freqs[mask]
    mask2 = freqs <= 800
    
    freqs = freqs[mask2]
    fft_s_amp_fem = fft_s_amp_fem[mask][mask2]
    fft_h_amp_fem = fft_h_amp_fem[mask][mask2]
    fft_s_amp_tot = fft_s_amp_tot[mask][mask2]
    fft_h_amp_tot = fft_h_amp_tot[mask][mask2]
    fft_h_test = fft_h_test[mask][mask2]
    fft_h_test_tot = fft_h_test_tot[mask][mask2]
    fft_s_test = fft_s_test[mask][mask2]
    fft_s_test_tot = fft_s_test_tot[mask][mask2]
    
    max_amp_h, max_amp_s, max_amp_test_h, max_amp_s_c, max_amp_h_c, max_amp_h_test, max_amp_s_test, max_amp_s_test_tot = np.max(fft_h_amp_tot), np.max(fft_s_amp_tot), np.max(fft_h_test_tot), np.max(fft_s_amp_fem), np.max(fft_h_amp_fem), np.max(fft_h_test), np.max(fft_s_test), np.max(fft_s_test_tot)
    fft_h_amp_tot = fft_h_amp_tot / max_amp_h
    fft_s_amp_tot = fft_s_amp_tot / max_amp_s
    fft_h_test_tot = fft_h_test_tot / max_amp_test_h
    fft_s_amp_fem = fft_s_amp_fem / max_amp_s_c
    fft_h_amp_fem = fft_h_amp_fem / max_amp_h_c
    fft_h_test = fft_h_test / max_amp_h_test
    fft_s_test = fft_s_test / max_amp_s_test
    fft_s_test_tot = fft_s_test_tot / max_amp_s_test_tot

    
    print("FFT calculees.")
    Title = "Spectre de la tension en sortie des micros\n(Fondamental à " + str(F) + " Hz)"
    print("Affichage des FFT...")
    
    #-- 01 --#
    mod.save_plot("Python/Exports/FFT/01-FFT_th.png",
            (freqs, fft_h_amp_fem, {'color': Color_h, 'label': "Micro Humbucker - Theorique", 'linestyle': ls_th}),
            (freqs, fft_s_amp_fem, {'color': Color_s, 'label': "Micro Singlecoil - Theorique", 'linestyle': ls_th}),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
    print("FFT sauvegardees sous Python/Exports/01-FFT_th.png.")
    
    mod.save_plot("Python/Exports/FFT/01-FFT_exp.png",
            (freqs, fft_s_test, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': ls_exp, 'color': Color_s_test, 'label': "Micro Singlecoil - Experimental"}),
            (freqs, fft_h_test, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': ls_exp, 'color': Color_h_test, 'label': "Micro Humbucker - Experimental"}),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")

    mod.save_plot("Python/Exports/FFT/01-FFT_s.png",
            (freqs, fft_s_test, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': ls_exp, 'color': Color_s_test, 'label': "Micro Singlecoil - Experimental"}),
            (freqs, fft_s_amp_fem, {'color': Color_s, 'label': "Micro Singlecoil - Theorique", 'linewidth':lw_th}),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
    print("FFT sauvegardees sous Python/Exports/01-FFT_s.png.")
    
    mod.save_plot("Python/Exports/FFT/01-FFT_h.png",
            (freqs, fft_h_test, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': ls_exp, 'color': Color_h_test, 'label': "Micro Humbucker - Experimental"}),
            (freqs, fft_h_amp_fem, {'color': Color_h, 'label': "Micro Humbucker - Theorique", 'linewidth':lw_th}),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
    print("FFT sauvegardees sous Python/Exports/01-FFT_h.png.")
    
    #-- 02 --#
    
    #-- 03 --#
    mod.save_plot("Python/Exports/FFT/03-FFT_th.png",
            (freqs, fft_h_amp_tot, {'color': Color_h, 'label': "Micro Humbucker - Theorique"}),
            (freqs, fft_s_amp_tot, {'color': Color_s, 'label': "Micro Singlecoil - Theorique"}),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
    print("FFT sauvegardees sous Python/Exports/03-FFT_th.png.")

    mod.save_plot("Python/Exports/FFT/03-FFT_s.png",
            (freqs, fft_s_test_tot, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': ls_exp, 'color': Color_s_test, 'label': "Micro Singlecoil - Experimental"}),
            (freqs, fft_s_amp_tot, {'color': Color_s, 'label': "Micro Singlecoil - Theorique", 'linewidth':lw_th}),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
    print("FFT sauvegardees sous Python/Exports/03-FFT_s.png.")

    mod.save_plot("Python/Exports/FFT/03-FFT_h.png",
            (freqs, fft_h_test_tot, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': ls_exp, 'color': Color_h_test, 'label': "Micro Humbucker - Experimental"}),
            (freqs, fft_h_amp_tot, {'color': Color_h, 'label': "Micro Humbucker - Theorique", 'linewidth':lw_th}),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
    print("FFT sauvegardees sous Python/Exports/03-FFT_h.png.")

    mod.save_plot("Python/Exports/FFT/03-FFT_exp.png",
            (freqs, fft_s_test_tot, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': ls_exp, 'color': Color_s_test, 'label': "Micro Singlecoil - Experimental"}),
            (freqs, fft_h_test_tot, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': ls_exp, 'color': Color_h_test, 'label': "Micro Humbucker - Experimental"}),
            title=Title, xlabel="Frequence [$Hz$]", ylabel="Tension normalisee")
    
print("-" * 100)
print("\nAffichage des courbes...")
#-- 01 --#

#* --- Attenuation --- *#
L_freqs, L_pic_s_exp = mod.extraire_pics_separes(freqs, fft_s_test)
L_freqs, L_pic_h_exp = mod.extraire_pics_separes(freqs, fft_h_test)

freqs = np.linspace(20, 20_000, 1000)
ratio_exp = 20*np.log10(L_pic_h_exp / L_pic_s_exp)
ratio_th, peigne = mod.attenuation_theorique_db(freqs, x1, x2, xm, L, F)

mod.save_plot("Python/Exports/01-Attenuation.png",
    (freqs, ratio_th, {'color': "#444444", 'label': "Gain Theorique [$dB$]", "linewidth": 1.5}),
    (L_freqs, ratio_exp, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': '', 'color': Color_h_test, 'label': "Gain experimentale [$dB$]"}),
    (freqs, peigne, {'color': Color_h, 'label': "Filtrage en peigne spatial [$dB$]", "linewidth": 2}),
    title=f"Influence de la position du micro sur le spectre\nFiltrage spatial pour un fondamental à {F}$Hz$", xlabel="Frequence [$Hz$]", ylabel="Gain relatif [$dB$]", log=True, vspan=True)

f_att_th = 440
f_att_exp = 470
eps_att = (f_att_th - f_att_exp) / f_att_th * 100

print(f"freq theorique : {f_att_th} Hz")
print(f"freq exp : {f_att_exp} Hz")
print(f"\nEcart a la theorie d'attenuation : {eps_att} %\n")

#* --- Comparaison des micros Theorique (total) ---*#
mod.save_plot("Python/Exports/Sig/01-fem_th.png",
        (temps, fem_h, {'color': Color_h, 'label': "Micro Humbucker - Theorique [$V$]"}),
        (temps, fem_s, {'color': Color_s, 'label': "Micro Singlecoil - Theorique [$V$]"}),
        title="Tension induite dans les micros\n(Fondamental à " + str(F) + " Hz)",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print("Courbes sauvegardees sous Python/Exports/01-fem_th.png.")
#* --- Comparaison des micros Theorique (corde seule) ---*#
mod.save_plot("Python/Exports/Sig/01-fem_singlecoil.png",
        (temps, Test_s, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': 'None', 'color': Color_s_test, 'label': "Micro Singlecoil - Experimental [$V$]"}),
        (temps, fem_s, {'color': Color_s, 'label': "Micro Singlecoil - Theorique [$V$]"}),
        title="Tension induite dans le micro Singlecoil\n(Fondamental à " + str(F) + " Hz)",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print("Courbes sauvegardees sous Python/Exports/01-fem_singlecoil.png.")
#* --- Micro Humbucker (corde) ---*#
mod.save_plot("Python/Exports/Sig/01-fem_humbucker.png",
        (temps, Test_h, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': 'None', 'color': Color_h_test, 'label': "Micro Humbucker - Experimental [$V$]"}),
        (temps, fem_h, {'color': Color_h, 'label': "Micro Humbucker - Theorique [$V$]"}),
        title="Tension induite dans le micro Humbucker\n(Fondamental à " + str(F) + " Hz)",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print("Courbes sauvegardees sous Python/Exports/01-fem_humbucker.png.")
#-- 02 --#

#-- 03 --#

#* --- Micro SingleCoil (Parasite) ---*#
mod.save_plot("Python/Exports/Sig/03-Parasite_singlecoil.png",
        (temps, Test_s_p[:4000], {'marker': m_exp, 'markersize': ms_exp, 'linestyle': 'None', 'color': Color_s_test, 'label': "Micro Singlecoil - Experimental [$V$]"}),
        (temps, fem_s_par, {'color': Color_s, 'label': "Micro Singlecoil - Theorique [$V$]"}),
        title="Tension induite dans le micro SingleCoil \nsoumis à un champs magnetique de 50µT",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]", ylim=(-0.5, 0.5))
print("Courbes sauvegardees sous Python/Exports/03-Parasite_singlecoil.png.")
#* --- Micro Humbucker (Parasite) ---*#
mod.save_plot("Python/Exports/Sig/03-Parasite_humbucker.png",
        (temps, Test_h_p[:4000], {'marker': m_exp, 'markersize': ms_exp, 'linestyle': 'None', 'color': Color_h_test, 'label': "Micro Humbucker - Experimental [$V$]"}),
        (temps, fem_h_par, {'color': Color_h, 'label': "Micro Humbucker - Theorique [$V$]"}),
        title="Tension induite dans le micro Humbucker \nsoumis à un champs magnetique de 50µT",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]", ylim=(-0.5, 0.5))
print("Courbes sauvegardees sous Python/Exports/03-Parasite_humbucker.png.")
#* --- Droite champ parasite ---*#
mod.save_plot("Python/Exports/03-Droite.png",
        (List_Bpar*10**6, List_amp_h_th*10**3, {'color': Color_h, 'label': "Micro Humbucker - Theorique [$V$]"}),
        (List_Bpar*10**6, List_amp_h_test*10**3, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': 'None', 'color': Color_h_test, 'label': "Micro Humbucker - Experimental [$V$]"}),
        (List_Bpar*10**6, List_amp_h_test_lin*10**3, {'linestyle': 'dashed', 'color': Color_h_test}),
        (List_Bpar*10**6, List_amp_s_th*10**3, {'color': Color_s, 'label': "Micro Singlecoil - Theorique [$V$]"}),
        (List_Bpar*10**6, List_amp_s_test*10**3, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': 'None', 'color': Color_s_test, 'label': "Micro Singlecoil - Experimental [$V$]"}),
        (List_Bpar*10**6, List_amp_s_test_lin*10**3, {'linestyle': 'dashed', 'color': Color_s_test}),
        title="Tension parasite induite en fonction du champ magnetique exterieur à 50 $Hz$",
        xlabel="Amplitude du champ parasite [$µT$]", ylabel="Ampltitude Tension [$mV$]")
print("Courbes sauvegardees sous Python/Exports/03-Droite.png.")
#* --- SingleCoil (total) ---*#
mod.save_plot("Python/Exports/Sig/03-Complet_singlecoil.png",
        (temps, Test_s_tot, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': 'None', 'color': Color_s_test, 'label': "Micro Singlecoil - Experimental [$V$]"}),
        (temps, fem_s_tot, {'color': Color_s, 'label': "Micro Singlecoil - Theorique [$V$]"}),
        title="Tension induite dans le micro Singlecoil - Total",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print("Courbes sauvegardees sous Python/Exports/03-Complet_singlecoil.png.")
#* --- Humbucker (total) ---*#
mod.save_plot("Python/Exports/Sig/03-Complet_humbucker.png",
        (temps, Test_h_tot, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': 'None', 'color': Color_h_test, 'label': "Micro Humbucker - Experimental [$V$]"}),
        (temps, fem_h_tot, {'color': Color_h, 'label': "Micro Humbucker - Theorique [$V$]"}),
        title="Tension induite dans le micro Humbucker - Total",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print("Courbes sauvegardees sous Python/Exports/03-Complet_humbucker.png.")

#* --- Comparaison des micros Theorique (total) ---*#
mod.save_plot("Python/Exports/Sig/03-Complet_th.png",
        (temps, fem_h_tot, {'color': Color_h, 'label': "Micro Humbucker - Theorique [$V$]"}),
        (temps, fem_s_tot, {'color': Color_s, 'label': "Micro Singlecoil - Theorique [$V$]"}),
        title="Tension induite dans les micros",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")
print("Courbes sauvegardees sous Python/Exports/03-Complet_th.png.")
#-- 99 --#
#* --- Total ---*#
mod.save_plot("Python/Exports/Sig/99-total.png",
        (temps, Test_s_tot, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': 'None', 'color': Color_s_test, 'label': "Micro Singlecoil - Experimental [$V$]"}),
        (temps, Test_h_tot, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': 'None', 'color': Color_h_test, 'label': "Micro Humbucker - Experimental [$V$]"}),
        (temps, fem_h_tot, {'color': Color_h, 'label': "Micro Humbucker - Theorique [$V$]"}),
        (temps, fem_s_tot, {'color': Color_s, 'label': "Micro Singlecoil - Theorique [$V$]"}),
        title="Tension induite dans les micros",
        xlabel="Temps [$s$]", ylabel="Tension [$V$]")

print("Courbes sauvegardees sous Python/Exports/99-total.png.")
print("-" * 100)
print("\nCalcul des sensibilites et CMRR theorique")
print(f"k_s = {ks*10**-3} mV/microTesla")
print(f"k_h = {kh*10**-3} mV/microTesla")
print(f"CMRR = { ks / kh }")
print(f"CMRR = {20*np.log10(ks / kh)} dB")
print("\nCalcul des sensibilites et CMRR experimental")
print(f"k_s_exp = {k_s_exp*10**-3} mV/microTesla")
print(f"k_h_exp = {k_h_exp*10**-3} mV/microTesla")
print(f"CMRR_exp = { k_s_exp / k_h_exp }")
print(f"CMRR_exp = {20*np.log10( k_s_exp / k_h_exp )} dB")
print("\nCalcul des erreurs relatives...")
eps_h = abs((k_h_exp - kh) / kh) * 100
eps_s = abs((k_s_exp - ks) / ks) * 100
eps = abs((ks/kh - k_s_exp/k_h_exp) / (ks/kh)) * 100
eps_cmrr = abs(20*np.log10(ks/kh) - 20*np.log10(k_s_exp/k_h_exp))/(20*np.log10(ks/kh)) *100

print(f"\nErreur relative k_h : {eps_h} %")
print(f"Erreur relative k_s : {eps_s} %")
print(f"Erreur relative gamma : {eps} %")
print(f"Erreur relative cmrr : {eps_cmrr} %")

print("-" * 100)
print("\n---\nFin de l'execution de Modele_fem.py\n---\n")