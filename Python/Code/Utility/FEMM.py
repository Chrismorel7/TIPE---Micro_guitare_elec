import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

# --- CONFIGURATION DU LISSAGE ---
# Choix : "AUCUN", "SAVGOL" (Recommande), ou "SPLINE"
METHODE_LISSAGE = "SAVGOL" 

# --- 1. Paramètres Physiques ---
freq = 78  # Frequence de la corde en Hz
omega = 2 * np.pi * freq
N_spires = 1 # Nombre de spires (Mettez votre valeur reelle, ex: 6000)

# --- 2. Chargement des donnees ---
try:
    data = np.loadtxt("Docs/FEMM/resultats_flux.txt", delimiter=",", skiprows=2)
except OSError:
    # Generation de donnees factices pour l'exemple si le fichier est absent
    print("Fichier introuvable, generation de donnees de test...")
    t_test = np.linspace(0, 1/freq, 200)
    angles = t_test * omega
    deplacement = np.sin(angles)
    # Flux avec bruit numerique simule
    flux = 3e-6 * (1 - 0.1*np.sin(angles)) + 1e-8 * np.random.normal(0, 1, 200)
else:
    angles = data[:, 0]      
    deplacement = data[:, 1] 
    flux = data[:, 2]        

# --- 3. Conversion Espace -> Temps ---
temps = angles / omega

# --- 4. Traitement et Lissage (eTAPE CLe) ---
flux_lisse = flux.copy()
dphi_dt = np.zeros_like(flux)

if METHODE_LISSAGE == "SAVGOL":
    # Filtre de Savitzky-Golay : Lisse en gardant la forme du signal
    # window_length : doit être impair. Plus il est grand, plus c'est lisse.
    # polyorder : ordre du polynôme (2 ou 3 est bien).
    window_length = 51 # A ajuster selon le nombre de points (ex: nb_points / 5)
    if window_length >= len(flux): window_length = len(flux) // 2 * 2 + 1
    
    flux_lisse = savgol_filter(flux, window_length=window_length, polyorder=3)
    dphi_dt = np.gradient(flux_lisse, temps)

elif METHODE_LISSAGE == "SPLINE":
    # Interpolation Spline : Cree une fonction mathematique lisse
    # s = facteur de lissage. Trop grand = perte de signal, Trop petit = garde le bruit.
    spl = UnivariateSpline(temps, flux, s=1e-15) 
    flux_lisse = spl(temps)
    dphi_dt = spl.derivative()(temps) # Derivee analytique très propre

else: # AUCUN
    flux_lisse = flux
    dphi_dt = np.gradient(flux, temps)

# --- 5. Calcul de la Tension ---
# e = - N * dPhi / dt
tension = -N_spires * dphi_dt
tension_mV = tension * 1000

# --- 6. Trace des courbes ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

# Graphique 1 : Vibration corde
ax1.plot(temps * 1000, deplacement, 'k--', alpha=0.5, label='Deplacement')
ax1.set_ylabel('Position (mm)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_title(f'Simulation FEMM (f = {freq} Hz) - Methode : {METHODE_LISSAGE}')
ax1.legend(loc='upper right')

# Graphique 2 : Flux Magnetique (Comparaison Brut vs Lisse)
ax2.plot(temps * 1000, flux * 1e6, 'b.', markersize=2, alpha=0.3, label='Flux Brut (Bruit numerique)')
ax2.plot(temps * 1000, flux_lisse * 1e6, 'g-', linewidth=2, label='Flux Lisse')
ax2.set_ylabel(r'Flux ($\mu$Wb)')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='upper right')

# Graphique 3 : Tension Induite
ax3.plot(temps * 1000, tension_mV, 'r-', linewidth=2, label=r'Tension $e(t)$')
ax3.set_ylabel('Tension (mV)')
ax3.set_xlabel('Temps (ms)')
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(loc='upper right')

# Calcul V_eff et V_max
v_max = np.max(np.abs(tension_mV))
v_rms = np.std(tension_mV)

info_text = f"Tension Max : {v_max:.2f} mV\nTension RMS : {v_rms:.2f} mV"
ax3.text(0.02, 0.05, info_text, transform=ax3.transAxes, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

plt.tight_layout()
plt.show()