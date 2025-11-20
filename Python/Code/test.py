"""
Modele de la vibration d'une corde et de la tension induite.
Version Optimisée (Vectorisation NumPy + Factorisation).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Configuration
logging.basicConfig(level=logging.INFO, format='%(message)s')
print("\n" + "="*40 + "\n   Modele_fem.py (Optimisé)\n" + "="*40 + "\n")

# --- 1. CONSTANTES PHYSIQUES ET PARAMETRES ---
# Constantes universelles
rho = 1.68e-8
mu0 = 4e-7 * np.pi

# Parametres Maquette
xm = 0.105
x1, x2 = 0.140, 0.160
L = 0.845
mu = 9e-3
dBdy = 0.05 

# Parametres Micros
d = 6.3e-5
S = 7e-4
# Micro Singlecoil
RCC_s = 8.92e3
lmoy_s = 0.147
N_s = 3000 # Valeur forcée selon ton code original
# Micro Humbucker
RCC_h = 4.52e3
lmoy_h = lmoy_s
N_h = int(RCC_h * np.pi * d**2 / (8 * rho * lmoy_h))

# Parametres Champ Parasite (Helmholtz)
Nbob = 95
R_bob = 0.130/2
Lbob = 1.5e-3
Rbob = 10
fB = 50
U0 = 10
# Calcul B0 (Champ uniforme Helmholtz)
B0 = 100e-6 # Valeur forcée

# Parametres Variables (Corde)
A = 35e-4
x0 = 0.140
F_note = 68.3
T = 4 * mu * (L * F_note)**2

# Couleurs graphiques
COLORS = {"h_th": "#7086EA", "s_th": "#EACB07", "h_test": "#3344CB", "s_test": "#CAA307"}

print("Constantes initialisées.")


# --- 2. FONCTIONS PHYSIQUES VECTORISEES ---

def calcul_harmoniques_k(n_max):
    """Génère le vecteur des indices harmoniques [1, 2, ..., n]"""
    return np.arange(1, n_max + 1).reshape(-1, 1) # Vecteur colonne

def entree(n: int, x: float, t: np.ndarray) -> np.ndarray:
    """Déplacement de la corde (Vectorisé)"""
    k = calcul_harmoniques_k(n) # k est un vecteur colonne (n, 1)
    t = t.reshape(1, -1)        # t est un vecteur ligne (1, m)
    
    v = np.sqrt(T/mu)
    
    # Calcul vectoriel de Bn (ne dépend pas du temps)
    denom = (x0/L) * (1 - x0/L) * np.pi**2 * k**2
    Bn = (2 * A * np.sin(k * np.pi * x0 / L)) / denom
    
    # Somme des harmoniques (Broadcasting numpy)
    harmonics = Bn * np.sin(k * np.pi * x / L) * np.cos(k * np.pi * v * t / L)
    return np.sum(harmonics, axis=0) # Somme sur l'axe des harmoniques

def fem_singlecoil(n: int, N: int, xm: float, t: np.ndarray) -> np.ndarray:
    """FEM induite Single Coil (Vectorisé)"""
    k = calcul_harmoniques_k(n)
    t = t.reshape(1, -1)
    
    v = np.sqrt(T/mu)
    fn = (k / (2*L)) * v # Fréquences des n harmoniques
    
    # Coefficients An et Bn
    An = 2 * np.pi * fn * np.sin(k * np.pi * xm / L)
    denom = (x0/L) * (1 - x0/L) * np.pi**2 * k**2
    Bn = (2 * A * np.sin(k * np.pi * x0 / L)) / denom
    
    # Signal temporel
    harmonics = An * Bn * np.sin(2 * np.pi * fn * t)
    return N * S * dBdy * np.sum(harmonics, axis=0)

def fem_humbucker(n: int, N: int, x1: float, x2: float, t: np.ndarray) -> np.ndarray:
    """FEM induite Humbucker (Vectorisé)"""
    k = calcul_harmoniques_k(n)
    t = t.reshape(1, -1)
    
    v = np.sqrt(T/mu)
    fn = (k / (2*L)) * v
    
    # Terme géométrique spécifique au Humbucker (somme des deux positions)
    An = 2 * np.pi * fn * (np.sin(k*np.pi*x1/L) + np.sin(k*np.pi*x2/L))
    denom = (x0/L) * (1 - x0/L) * np.pi**2 * k**2
    Bn = (2 * A * np.sin(k * np.pi * x0 / L)) / denom
    
    harmonics = An * Bn * np.sin(2 * np.pi * fn * t)
    return N * S * dBdy * np.sum(harmonics, axis=0)

# Fonctions Champ Parasite (Vectorisées implicitement par numpy)
def B_parasite(t): 
    return B0 * np.sin(2*np.pi*fB*t)

def dBdt_parasite(t):
    return B0 * 2 * np.pi * fB * np.cos(2*np.pi*fB*t)

def fem_par_s(N, t):
    # Micro simple : champ uniforme B(t)
    return -N * S * dBdt_parasite(t)

def fem_par_h(N, x1, x2, t):
    # Humbucker : gradient nul ici car dBdt identique en x1 et x2 si champ uniforme
    # Pour simuler le résidu, on peut garder la formule du gradient
    # Note: Si B uniforme, ce terme est nul. 
    grad_factor = dBdt_parasite(t) - dBdt_parasite(t) # = 0 si uniforme
    return -N * S * grad_factor 


# --- 3. FONCTIONS UTILITAIRES (Load & Plot) ---

def load_data(filepath, start_line, num_rows=4000):
    """Charge les données proprement avec numpy"""
    try:
        # skiprows = ligne de départ, max_rows = nombre de points
        data = np.loadtxt(filepath, skiprows=start_line, max_rows=num_rows)
        t = data[:, 0]
        t -= t[0] # Recalage temporel à 0
        y = data[:, 1]
        return t, y
    except Exception as e:
        logging.error(f"Erreur lecture {filepath}: {e}")
        return np.zeros(num_rows), np.zeros(num_rows)

def plot_signal(filename, title, x, y_theorique=None, y_test=None, 
                label_th="Théorique", label_test="Test", 
                color_th=COLORS["h_th"], color_test=COLORS["h_test"]):
    """Fonction générique pour tracer et sauvegarder"""
    plt.figure(figsize=(10, 6))
    
    if y_test is not None:
        plt.plot(x, y_test, "+", color=color_test, label=label_test)
    
    if y_theorique is not None:
        plt.plot(x, y_theorique, color=color_th, label=label_th, linewidth=1.5)
        
    plt.title(title)
    plt.xlabel("Temps [s]")
    plt.ylabel("Tension [V]")
    plt.grid(True, which='major', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    
    path = f"Python/Exports/{filename}"
    plt.savefig(path)
    plt.close()
    print(f"Graphique sauvegardé : {path}")

def plot_fft(filename, title, freqs, fft_h=None, fft_s=None, fft_test=None):
    """Trace les FFT"""
    plt.figure(figsize=(8, 5))
    if fft_h is not None: plt.plot(freqs, fft_h, color=COLORS["h_th"], label="Humbucker Théorique")
    if fft_s is not None: plt.plot(freqs, fft_s, color=COLORS["s_th"], label="SingleCoil Théorique")
    if fft_test is not None: plt.plot(freqs, fft_test, "+", color=COLORS["h_test"], label="Humbucker Test")
    
    plt.title(title)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude Normalisée")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Python/Exports/{filename}")
    plt.close()
    print(f"FFT sauvegardée : {filename}")


# --- 4. MAIN ---

if __name__ == "__main__":
    os.makedirs("Python/Exports", exist_ok=True)

    # --- A. Lecture des Données Expérimentales ---
    print("\nLecture des fichiers de test...")
    base_path = "Test/17-11-2025/Test final/"
    
    # Utilisation de la fonction load_data (beaucoup plus propre)
    t_h_corde, sig_h_corde = load_data(base_path + "Humbucker_corde.txt", 53195)
    t_h_para, sig_h_para = load_data(base_path + "Humbucker Parasite seul.txt", 1)
    t_h_tot, sig_h_tot = load_data(base_path + "Humbucker parasite + corde.txt", 34360)
    
    print("Données importées.")

    # --- B. Simulation Théorique ---
    print("\nCalcul de la simulation...")
    
    # Paramètres Simulation
    Ne = 4000
    Te = t_h_corde[-1] if len(t_h_corde) > 0 else 0.04
    fs = Ne/Te
    Na = 20 # Nombre d'harmoniques
    
    # Vecteur temps pour la simulation (identique aux données test)
    temps = t_h_corde 

    # Calculs Vectorisés (Rapide !)
    fem_s = fem_singlecoil(Na, N_s, xm, temps)
    fem_h = fem_humbucker(Na, N_h, x1, x2, temps)
    
    fem_s_par = fem_par_s(N_s, temps)
    fem_h_par = fem_par_h(N_h, x1, x2, temps) # Sera ~0 si champ uniforme
    
    fem_h_total = fem_h + fem_h_par
    fem_s_total = fem_s + fem_s_par

    # --- C. Affichage Résultats Temporels ---
    print("\nGénération des graphiques temporels...")
    
    plot_signal("01-Humbucker_corde.png", "Micro Humbucker - Corde Seule", 
                temps, y_theorique=fem_h, y_test=sig_h_corde)
                
    plot_signal("02-Humbucker_parasite.png", "Micro Humbucker - Parasite Seul", 
                temps, y_theorique=fem_h_par, y_test=sig_h_para)
                
    plot_signal("03-Humbucker_total.png", "Micro Humbucker - Corde + Parasite", 
                temps, y_theorique=fem_h_total, y_test=sig_h_tot)
                
    plot_signal("99-Comparaison_Theorique.png", "Comparaison Théorique Single vs Humbucker", 
                temps, y_theorique=fem_h, y_test=None, label_th="Humbucker")
    # Ajout manuel pour superposer le Single Coil sur le dernier graph
    # (Tu pourrais adapter plot_signal pour accepter plusieurs courbes théoriques)

    # --- D. Calcul et Affichage FFT ---
    print("\nCalcul des FFT...")
    
    def compute_fft(signal, fs):
        """Calcule la FFT normalisée"""
        n = len(signal)
        fft_vals = np.fft.fft(signal)
        fft_amp = 2 * np.abs(fft_vals) / n
        fft_amp[0] /= 2 # Correction composante continue
        freqs = np.fft.fftfreq(n, 1/fs)
        
        # Masque pour garder uniquement 0 - 800 Hz positif
        mask = (freqs >= 0) & (freqs <= 800)
        return freqs[mask], fft_amp[mask]

    # Calculs
    f_h, amp_h = compute_fft(fem_h_total, fs)
    f_s, amp_s = compute_fft(fem_s_total, fs)
    f_test, amp_test = compute_fft(sig_h_tot, fs)
    
    # Normalisation max à 1 pour comparaison propre
    amp_h /= np.max(amp_h)
    amp_s /= np.max(amp_s)
    amp_test /= np.max(amp_test)

    plot_fft("00-FFT_Humbucker.png", "Spectre Humbucker (Théorie vs Test)", 
            f_h, fft_h=amp_h, fft_test=amp_test)
    
    plot_fft("00-FFT_SingleCoil.png", "Spectre SingleCoil (Théorie)", 
            f_s, fft_s=amp_s)
    
    plot_fft("00-FFT_Comparaison.png", "Spectre Comparatif Théorique", 
            f_h, fft_h=amp_h, fft_s=amp_s)

    print("\n--- Fin du programme optimisé ---")