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

# --- Importations des bibliothèques ---
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


# --- Variables ---
Rs_s = 5.26e3           # Resistance DC SingleCoil [Ohms]
Rs_hp = 4.52e3          # Resistance DC Humbucker parallèle [Ohms]
Rs_hs = 15.34e3         # Resistance DC Humbucker série [Ohms]
Ramp = 1e6              # Resistance de l'ampli [Ohms]
R = 97.6e3              # Resistance experimentale du PDT [Ohms]
X0 = np.array([2.5, 0.5, 450])    # Initalisation [L, C, R_p]
LR = 0.001              # Taux d'apprentissage

u_Ugbf = 0.01  # Incertitude sur tension GBF [V] (±1% lecture + résolution ADC)
u_res = 0.01   # Incertitude sur tension mesurée [V]
u_R = 97.6e3 * 0.01  # Incertitude sur R : ±1% (spec résistance de précision)

# --- Graphique ---
kw_Hp_th = {'color': '#0000FF', 'label': 'Humbucker parallèle (Ajustement)', 'linewidth': 3, 'linestyle':(0, (1,1))}
kw_S_th = {'color': '#FF0000', 'label': 'SingleCoil (Ajustement)', 'linewidth': 3, 'linestyle':(0, (1,1))}
kw_Hs_th = {'color': '#009900', 'label': 'Humbucker serie (Ajustement)', 'linewidth': 3, 'linestyle':(0, (1,1))}
kw_Hp_exp = {'color': '#6666FF', 'label': 'Humbucker parallèle (Expérience)', 'marker':"o", 'markersize':6, 'linestyle':'None'}
kw_Hs_exp = {'color': '#66CC66', 'label': 'Humbucker serie (Expérience)', 'marker':"o", 'markersize':6, 'linestyle':'None'}
kw_S_exp = {'color': '#FF6666', 'label': 'SingleCoil (Expérience)', 'marker':"o", 'markersize':6, 'linestyle':'None'}


# --- Fonction pour lire les fichier experimentaux ---
def lire_fichier(PATH):
    f = []
    Vin = []
    Vs = []
    Vhp = []
    Vhs = []
    file = open(PATH)
    L_lignes = file.readlines()
    for ligne in L_lignes[1:]:
        ligne = ligne.strip().split(',')
        ligne = [x.replace('\n', '') for x in ligne]
        ligne = [x.replace(',', '.') for x in ligne]
        Vin.append(float(ligne[0]))
        f.append(float(ligne[1]))
        Vs.append(float(ligne[2]))
        Vhp.append(float(ligne[3]))
        Vhs.append(float(ligne[4]))
    return np.array(Vin), np.array(f), np.array(Vs), np.array(Vhp), np.array(Vhs)

# --- Fonction calculer les incertidude absolue ---
def calcul_incertitude_Z(Ra, u_Ra, U, u_U, V, u_V):
    """
    Calcule l'incertitude ABSOLUE sur Z [Ohm].
    Formule : Z = R * (U_gbf / U_res - 1)
    Propagation d'erreur :
        u(Z)/Z = sqrt[(u_R/R)² + (u_Ugbf/U_gbf)² + (u_ures/U_res)²]
    """    
    # Incertitude relative
    u_Z_rel = np.sqrt((u_Ra / Ra)**2 + 
                    (u_U / U)**2 + 
                    (u_V / V)**2)
    # Impédance pour chaque point
    Z_calc = Ra * (U / V - 1)
    # Incertitude absolue [Ohm]
    u_Z_abs = Z_calc * u_Z_rel
    return u_Z_abs

# --- Lectures des données experimentales ---
f, Vin, Vs, Vhp, Vhs = lire_fichier("Python/data/raw/01-Reponse_elec/Mesure_Impedance_PDT_100k.csv")

# --- Pont diviseur de tension pour isoler |Z| ---
Z_s = R * (21.1/Vs - 1)
Z_hp = R * (21.1/Vhp - 1)
Z_hs = R * (21.1/Vhs - 1)

u_Z_s = calcul_incertitude_Z(R, u_R, 21.1, u_Ugbf, Vs, u_res) * 1e-3  # kOhm
u_Z_hp = calcul_incertitude_Z(R, u_R, 21.1, u_Ugbf, Vhp, u_res) * 1e-3  # kOhm
u_Z_hs = calcul_incertitude_Z(R, u_R, 21.1, u_Ugbf, Vhs, u_res) * 1e-3  # kOhm
kw_S_exp['yerr'] = u_Z_s
kw_Hp_exp['yerr'] = u_Z_hp
kw_Hs_exp['yerr'] = u_Z_hs

# --- Affichage des incertitudes moyennes ---
print("\n" + "="*60)
print("INCERTITUDES SUR L'IMPEDANCE [kOhm]")
print("="*60)
print(f"SingleCoil:")
print(f"  - Incertitude moyenne: {np.mean(u_Z_s):.4f} kOhm")
print(f"  - Incertitude min/max: {np.min(u_Z_s):.4f} / {np.max(u_Z_s):.4f} kOhm")
print(f"  - Incertitude relative: {np.mean(u_Z_s / (Z_s*1e-3))*100:.2f}%")

print(f"\nHumbucker parallèle:")
print(f"  - Incertitude moyenne: {np.mean(u_Z_hp):.4f} kOhm")
print(f"  - Incertitude min/max: {np.min(u_Z_hp):.4f} / {np.max(u_Z_hp):.4f} kOhm")
print(f"  - Incertitude relative: {np.mean(u_Z_hp / (Z_hp*1e-3))*100:.2f}%")

print(f"\nHumbucker série:")
print(f"  - Incertitude moyenne: {np.mean(u_Z_hs):.4f} kOhm")
print(f"  - Incertitude min/max: {np.min(u_Z_hs):.4f} / {np.max(u_Z_hs):.4f} kOhm")
print(f"  - Incertitude relative: {np.mean(u_Z_hs / (Z_hs*1e-3))*100:.2f}%")
print("="*60 + "\n")


# --- Appel de l'algorithme de moindre-carre ---
L_S = np.zeros((3,50))
for i in range(1,51):
    X_s, S_s = mod.moindres_carres(X0, Rs_s, f, Z_s, i, LR)
    X_hp, S_hp = mod.moindres_carres(X0, Rs_hp, f, Z_hp, i, LR)
    X_hs, S_hs = mod.moindres_carres(X0, Rs_hs, f, Z_hs, i, LR)
    L_S[0, i-1] = S_s
    L_S[1, i-1] = S_hp
    L_S[2, i-1] = S_hs
L_S[0] = mod.normalize_signal(L_S[0])
L_S[1] = mod.normalize_signal(L_S[1])
L_S[2] = mod.normalize_signal(L_S[2])

print(f"SingleCoil: L={X_s[0]:.2f}H, C={X_s[1]:.2f}nF, Rp={X_s[2]:.1f}kOhm")
print(f"Humbucker parallele: L={X_hp[0]:.2f}H, C={X_hp[1]:.2f}nF, Rp={X_hp[2]:.1f}kOhm")
print(f"Humbucker série: L={X_hs[0]:.2f}H, C={X_hs[1]:.2f}nF, Rp={X_hs[2]:.1f}kOhm")

# --- Modèle numérique ---
freq = np.linspace(10, 30_000, 1000)
gain_s, phase_s = mod.calcul_bode(X_s, Rs_s, freq, Ramp)
gain_hp, phase_hp = mod.calcul_bode(X_hp, Rs_hp, freq, Ramp)
gain_hs, phase_hs = mod.calcul_bode(X_hs, Rs_hs, freq, Ramp)

# -- Courbe Convergence Algo ---
mod.save_plot("Python/Exports/01-Reponse_electrique_frequetielle/Convergence.png",
            (range(1,51), L_S[0]*100, kw_S_th),
            (range(1,51), L_S[2]*100, kw_Hs_th),
            (range(1,51), L_S[1]*100, kw_Hp_th),
            title="Convergence de l'algorithme de moindre-carre",
            xlabel="Nombre d'iteration [.]",
            ylabel="Image de l'erreur [%]")


# --- Diagramme de Bode ---
plt.figure(figsize=(8,6))
plt.subplot(2, 1, 1)
plt.axvspan(82.41, 1174.66, color="#ADD8E6", alpha=0.6, label="Frequences fondamentales", zorder=0)
plt.axvspan(1174.66, 5*1174.66, facecolor="none", alpha=0.5, hatch="////", 
                        edgecolor="#424242", label="Frequences harmoniques", zorder=0)
plt.plot(freq, gain_s, **kw_S_th)
plt.plot(freq, gain_hs, **kw_Hs_th)
plt.plot(freq, gain_hp, **kw_Hp_th)
plt.xscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylabel("Gain [$dB$]")
plt.title("Diagramme de Bode des micros")
plt.xlabel("Frequence [$Hz$]")
plt.xlim((10, 30000))

plt.subplot(2, 1, 2)
plt.axvspan(82.41, 1174.66, color="#ADD8E6", alpha=0.6, label="Frequences fondamentales", zorder=0)
plt.axvspan(1174.66, 5*1174.66, facecolor="none", alpha=0.5, hatch="////", 
                        edgecolor="#424242", label="Frequences harmoniques", zorder=0)
plt.plot(freq, phase_s, **kw_S_th)
plt.plot(freq, phase_hs, **kw_Hs_th)
plt.plot(freq, phase_hp, **kw_Hp_th)
plt.xscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylabel("Phase [$°$]")
plt.xlabel("Frequence [$Hz$]")
plt.xlim((10, 30000))
plt.legend(loc='lower left')
plt.savefig("Python/Exports/01-Reponse_electrique_frequetielle/00-Bode.png")
plt.close()

mod.save_plot("Python/Exports/01-Reponse_electrique_frequetielle/01-Gain.png",
                (freq, gain_s, kw_S_th),
                (freq, gain_hs, kw_Hs_th),
                (freq, gain_hp, kw_Hp_th),
                xlabel="Frequence $[Hz]$",
                ylabel="Gain $[dB]$",
                vspan=True,
                legend=True,
                xlim=(30, 20000),
                title="Diagramme de Bode des micros",
                log=True)


# --- Courbe Z = f(w) ---
mod.save_plot("Python/Exports/01-Reponse_electrique_frequetielle/Impedance.png",
                (freq, mod.Z_micro(X_s, Rs_s, 2*np.pi*freq)*1e-3, kw_S_th),
                (f, Z_s*1e-3, kw_S_exp),
                (freq, mod.Z_micro(X_hs, Rs_hs, 2*np.pi*freq)*1e-3, kw_Hs_th),
                (f, Z_hs*1e-3, kw_Hs_exp),
                (freq, mod.Z_micro(X_hp, Rs_hp, 2*np.pi*freq)*1e-3, kw_Hp_th),
                (f, Z_hp*1e-3, kw_Hp_exp),
                title=r"Impedances des micros $|Z|$ en fonction de la frequence",
                xlabel="Frequence [$Hz$]",
                ylabel=r"Impedance [$k\Omega$]",
                vspan=True,
                xlim=(10, 30_000),
                legend=True,
                error_bars=True)