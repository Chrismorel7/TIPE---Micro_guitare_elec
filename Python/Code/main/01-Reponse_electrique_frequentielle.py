"""Module de simulation electronique des micros de guitare électrique.

Ce module implémente les modèles electriques equivalents pour 
comparer les performances des micros SingleCoil et Humbucker. 
Il repose sur l'analyse du filtrage électrique interne RLC.

Calculs principaux :
    - Impédance complexe |Z(jw)| du circuit équivalent.
    -  Gain et phase d'un micro chargé H(jw)

Auteur: Christophe Morel.
Établissement: Lycée Ferdinand Buisson, CPGE PT.
Projet: TIPE N°39662 - Session 2026.
"""

                                            #!####################!#
                                            #!   Bibliothèques    !#
                                            #!####################!#

print("\nImportation des bibliotheques...")
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from library import Modelisation as mod
import numpy as np
import matplotlib.pyplot as plt
import logging
plt.close("all")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
print("Bibliotheques importees.")

                                            #!####################!#
                                            #!     Variables      !#
                                            #!####################!#

print("\nInitialisation des variables...")
Rs_s = 5.26e3           # Resistance DC SingleCoil [Ohms]
Rs_hp = 4.52e3          # Resistance DC Humbucker parallèle [Ohms]
Rs_hs = 15.10e3         # Resistance DC Humbucker série [Ohms]
Ramp = 1e6              # Resistance de l'ampli [Ohms]
X0 = np.array([2, 0.5, 100])    # Initalisation [L, C, R_p]
LR = 0.01               # Taux d'apprentissage

# --- Graphique ---
kw_Hp_th = {'color': '#000080', 'label': 'Humbucker parallèle - Théorie', 'linewidth': 2}
kw_S_th = {'color': '#8B0000', 'label': 'SingleCoil - Théorie', 'linewidth': 2}
kw_Hs_th = {'color': '#006400', 'label': 'Humbucker série - Théorie', 'linewidth': 2}
kw_Hp_exp = {'color': '#9400D3', 'label': 'Humbucker parallèle - experience', 'marker':"o", 'markersize':5, 'linestyle':':'}
kw_Hs_exp = {'color': '#00CD00', 'label': 'Humbucker série - experience', 'marker':"o", 'markersize':5, 'linestyle':':'}
kw_S_exp = {'color': '#FF8C00', 'label': 'SingleCoil - experience', 'marker':"o", 'markersize':5, 'linestyle':':'}


# --- Fonction pour lire les fichier experimentaux ---
def lire_fichier(PATH):
    f = []
    Vin = []
    Vs = []
    Vh = []
    file = open(PATH)
    L_lignes = file.readlines()
    
    for ligne in L_lignes[1:]:
        ligne = ligne.strip().split(',')
        ligne = [x.replace('\n', '') for x in ligne]
        ligne = [x.replace(',', '.') for x in ligne]
        Vin.append(float(ligne[0]))
        f.append(float(ligne[1]))
        Vs.append(float(ligne[2]))
        Vh.append(float(ligne[3]))
    
    return np.array(Vin), np.array(f), np.array(Vs), np.array(Vh)

print("Variables initialisees.")

                                            #!####################!#
                                            #! Lectures Fichiers  !#
                                            #!####################!#

f, Vin, Vs, Vhp = lire_fichier("Python/data/raw/!!Mesures_impedance_500Ohm_2_fev.csv")
# --- Pont diviseur de tension pour isoler |Z| ---
Z_s = 100000 * (21.1/Vs - 1)
Z_hp = 100000 * (21.1/Vhp - 1)

                                            #!####################!#
                                            #!      Fonction      !#
                                            #!####################!#

X_s, S_s = mod.Levenberg_Marquardt(X0, Rs_s, f, Z_s, 20, LR)
X_hp, S_hp = mod.Levenberg_Marquardt(X0, Rs_hp, f, Z_hp, 20, LR)
print(f"SingleCoil: L={X_s[0]:.2f}H, C={X_s[1]:.2f}nF, Rp={X_s[2]:.1f}kOhm")
print(f"Humbucker parallele: L={X_hp[0]:.2f}H, C={X_hp[1]:.2f}nF, Rp={X_hp[2]:.1f}kOhm")

freq = np.linspace(10, 100_000, 1000)
gain_s, phase_s = mod.calcul_bode(X_s, Rs_s, freq, Ramp)
gain_hp, phase_hp = mod.calcul_bode(X_hp, Rs_hp, freq, Ramp)

                                            #!####################!#
                                            #!     Affichage      !#
                                            #!####################!#

# --- Diagramme de Bode ---
print("\nAffichage du diagramme de Bode...")

plt.figure(figsize=(8,6))
plt.subplot(2, 1, 1)
plt.axvspan(82.41, 1174.66, color="#F0E68C", alpha=0.3, label="Frequences fondamentales", zorder=0)
plt.axvspan(1174.66, 5*1174.66, facecolor="none", alpha=0.5, hatch="////", 
                        edgecolor="#424242", label="Frequences harmoniques", zorder=0)
plt.plot(freq, gain_s, **kw_S_th)
plt.plot(freq, gain_hp, **kw_Hp_th)
plt.xscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylabel("Gain [$dB$]")
plt.title("Diagramme de Bode des micros")
plt.xlabel("Frequence [$Hz$]")
plt.xlim((10, 30000))

plt.subplot(2, 1, 2)
plt.axvspan(82.41, 1174.66, color="#F0E68C", alpha=0.3, label="Frequences fondamentales", zorder=0)
plt.axvspan(1174.66, 5*1174.66, facecolor="none", alpha=0.5, hatch="////", 
                        edgecolor="#424242", label="Frequences harmoniques", zorder=0)
plt.plot(freq, phase_s, **kw_S_th)
plt.plot(freq, phase_hp, **kw_Hp_th)
plt.xscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylabel("Phase [$°$]")
plt.xlabel("Frequence [$Hz$]")
plt.xlim((10, 30000))
plt.legend(loc='lower left')
plt.savefig("Python/Exports/01-Reponse_electrique_frequetielle/00-Bode.png")
plt.close()

# --- Courbe Z = f(w) ---
mod.save_plot("Python/Exports/01-Reponse_electrique_frequetielle/Impedance.png",
                (f, Z_s*1e-3, kw_S_exp),
                (f, Z_hp*1e-3, kw_Hp_exp),
                (freq, mod.Z_micro(X_s, Rs_s, 2*np.pi*freq)*1e-3, kw_S_th),
                (freq, mod.Z_micro(X_hp, Rs_hp, 2*np.pi*freq)*1e-3, kw_Hp_th),
                title=r"Impedances des micros $|Z|$ en fonction de la frequence",
                xlabel="Frequence [$Hz$]",
                ylabel=r"Impedance [$k\Omega$]",
                vspan=True,
                xlim=(10, 100_000),
                log=True,
                legend=True)

print("\n---\nFin de l'execution de Modele_elec.py\n---\n")