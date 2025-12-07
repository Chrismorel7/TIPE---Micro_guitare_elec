import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres Réels (À mesurer sur votre guitare) ---
L_corde = 0.648
f0 = 82.4 # Mi grave
v = 2 * L_corde * f0

# Positions des micros (Distance par rapport au chevalet en mètres)
# EXEMPLE : Mesurez ces valeurs sur votre guitare !
x_single_coil = 0.105  # ex: Micro milieu
x_humbucker   = 0.15  # ex: Micro manche
delta_x       = 0.018 # Largeur Humbucker

freqs = np.linspace(10, 20_000, 1000)
k = 2 * np.pi * freqs / v

# --- Modèle Complet ---
# Signal SC (Seulement position)
signal_sc = np.abs(np.sin(k * x_single_coil))

# Signal HB (Position + Filtre Peigne)
# Note : x_humbucker est le centre du micro
filtre_peigne = np.cos(k * delta_x / 2)
signal_hb = 2 * np.abs(np.sin(k * x_humbucker)) * np.abs(filtre_peigne)

# Ratio Théorique (Avec gestion de la division par zéro)
ratio = np.divide(signal_hb, signal_sc, out=np.zeros_like(signal_sc), where=signal_sc!=0)
ratio_db = 20 * np.log10(ratio + 1e-6)

# --- 3. Calcul du 1er "Trou" Théorique ---
f_notch = v / (2 * delta_x)
print(f"Fréquence d'annulation théorique : {f_notch:.2f} Hz")

# --- 4. Tracé ---
plt.figure(figsize=(5, 4))

# Courbe de gain
plt.semilogx(freqs, ratio_db, label="Filtrage spatial réel", color='grey', linewidth=0.5)
plt.semilogx(freqs, 20*np.log10(np.abs(np.cos(k*delta_x/2))), color="navy", label="Filtrage en peigne", linewidth=2)

# Marquer l'annulation
#plt.axvline(f_notch, color='red', linestyle='--', label=f'Annulation théorique ({int(f_notch)} Hz)')

# Esthétique
plt.title(f"Filtrage en Peigne Spatial du Humbucker\n(fondamental : {f0} Hz, $\Delta x$={delta_x*1000}mm)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Gain normalisé dB")
plt.grid(True, which='both', linestyle='--', alpha=0.7)

plt.axvspan(70, 1200, color='green', alpha=0.1, label="Zone accessible par fondamental")
plt.axvspan(1200, 5*1200, color='orange', alpha=0.1, label="Zone accessible par harmonique")
plt.legend()

plt.show()