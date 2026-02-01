"""Modèle electrique du micro de guitare electrique et affichage de la reponse en frequence via un diagramme de Bode.
    """

print("\n\n---\nModele_elec.py\n---\n")

                                            #!####################!#
                                            #!   Bibliothèques    !#
                                            #!####################!#

print("\nImportation des bibliotheques...")
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
Rs = 5.3e3          # Resistance du single coil en ohms
Rh = 4.52e3        # Resistance du humbucker en ohms
Ramp = 1e6         # Resistance de l'ampli en ohms
X0 = np.array([2, 0.5, 100])
LR = 0.01
Color_c = "#505050"
Color_h_test = "#008B8B"
Color_s_test = "#FF8C00"
ls_exp=':'
lw_exp=2
m_exp="+"
ms_exp=10
Color_h = "#000080"
Color_s = "#8B0000"
ls_th='-'
lw_th=2

# --- 2. Fonction de Transfert (Modèle Passe-Bas 2nd Ordre) ---
def lire_fichier(PATH):
    f = []
    Vin = []
    Vs = []
    Vh = []
    file_s2 = open(PATH)
    L_lignes = file_s2.readlines()
    
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

Vin, f, Vs, Vh = lire_fichier("Python/data/raw/!Mesure_Impedance.csv")

Z_sc = Ramp * (Vs / (24000 - Vs))
Z_hb = Ramp * (Vh / (24000 - Vh))

                                            #!####################!#
                                            #!      Fonction      !#
                                            #!####################!#

# --- Execution ---
# Initialisation suggeree : L=2H, C=0.5nF, Rp=300kOhm

Liste_S_sc = []
Liste_S_hb = []
liste_iter = []
Liste_Xsc = []
Liste_Xhb =[]
for i in range(1, 50):
    X_sc, S_sc = mod.Levenberg_Marquardt(X0, Rs, f, Z_sc, i, LR)
    X_hb, S_hb = mod.Levenberg_Marquardt(X0, Rh, f, Z_hb, i, LR)
    Liste_S_sc.append(S_sc)
    Liste_S_hb.append(S_hb)
    Liste_Xsc.append(X_sc)
    Liste_Xhb.append(X_hb)
    liste_iter.append(i)

Liste_S_sc = np.array(Liste_S_sc)
Liste_S_hb = np.array(Liste_S_hb)

X_sc, S_sc = mod.Levenberg_Marquardt(X0, Rs, f, Z_sc, 20, LR)
X_hb, S_hb = mod.Levenberg_Marquardt(X0, Rh, f, Z_hb, 20, LR)

print(f"SingleCoil: L={X_sc[0]:.2f}H, C={X_sc[1]:.2f}nF, Rp={X_sc[2]:.1f}kOhm")
print(f"Humbucker: L={X_hb[0]:.2f}H, C={X_hb[1]:.2f}nF, Rp={X_hb[2]:.1f}kOhm")

                                            #!####################!#
                                            #!     Affichage      !#
                                            #!####################!#

print("\nAffichage du diagramme de Bode...")

plt.figure(figsize=(8,6))
plt.subplot(2, 1, 1)
plt.axvspan(82.41, 1174.66, color="#F0E68C", alpha=0.3, label="Frequences fondamentales", zorder=0)
plt.axvspan(1174.66, 5*1174.66, facecolor="none", alpha=0.5, hatch="////", 
                        edgecolor="#424242", label="Frequences harmoniques", zorder=0)
plt.plot(np.linspace(20, 20000, 1000), mod.calcul_bode(X_sc, Rs, np.linspace(20, 20000, 1000), Ramp)[0], linestyle=ls_th, linewidth=lw_th, color=Color_s, label="Micro SingleCoil - Theorie")
plt.plot(np.linspace(20, 20000, 1000), mod.calcul_bode(X_hb, Rh, np.linspace(20, 20000, 1000), Ramp)[0], linestyle=ls_th, linewidth=lw_th, color=Color_h, label="Micro Humbucker - Theorie")
plt.xscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylabel("Gain [$dB$]")
plt.title("Diagramme de Bode des micros")
plt.xlabel("Frequence [$Hz$]")
plt.xlim((0, 20000))

plt.subplot(2, 1, 2)
plt.axvspan(82.41, 1174.66, color="#F0E68C", alpha=0.3, label="Frequences fondamentales", zorder=0)
plt.axvspan(1174.66, 5*1174.66, facecolor="none", alpha=0.5, hatch="////", 
                        edgecolor="#424242", label="Frequences harmoniques", zorder=0)
plt.plot(np.linspace(20, 20000, 1000), mod.calcul_bode(X_sc, Rs, np.linspace(20, 20000, 1000), Ramp)[1], linestyle=ls_th, linewidth=lw_th, color=Color_s, label="Micro SingleCoil - Theorie")
plt.plot(np.linspace(20, 20000, 1000), mod.calcul_bode(X_hb, Rh, np.linspace(20, 20000, 1000), Ramp)[1], linestyle=ls_th, linewidth=lw_th, color=Color_h, label="Micro Humbucker - Theorie")
plt.xscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylabel("Phase [$°$]")
plt.xlabel("Frequence [$Hz$]")
plt.xlim((0, 20000))
plt.legend(loc='lower left')
plt.savefig("Python/Exports/01-Reponse_electrique_frequetielle/00-Bode.png")
plt.close()

mod.save_plot("Python/Exports/01-Reponse_electrique_frequetielle/Impedance.png",
                (f, Z_sc*1e-3, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': ls_exp, 'linewidth': lw_exp, 'color': Color_s_test, 'label': "Micro SingleCoil"}),
                (f, Z_hb*1e-3, {'marker': m_exp, 'markersize': ms_exp, 'linestyle': ls_exp, 'linewidth': lw_exp, 'color': Color_h_test, 'label': "Micro Humbucker"}),
                (np.linspace(20, 20000, 1000), mod.Z_micro(X_sc, Rs, 2*np.pi*np.linspace(20, 20000, 1000))*1e-3, {'linestyle': ls_th, 'linewidth': lw_th, 'color': Color_s, 'label': "Micro SingleCoil - Theorie"}),
                (np.linspace(20, 20000, 1000), mod.Z_micro(X_hb, Rh, 2*np.pi*np.linspace(20, 20000, 1000))*1e-3, {'linestyle': ls_th, 'linewidth': lw_th, 'color': Color_h, 'label': "Micro Humbucker - Theorie"}),
                title=r"Impedances des micros $|Z|$ en fonction de la frequence",
                xlabel="Frequence [$Hz$]",
                ylabel=r"Impedance [$k\Omega$]",
                vspan=True,
                xlim=(20, 20000),
                legend=True)

mod.save_plot("Python/Exports/01-Reponse_electrique_frequetielle/Convergence_algo.png",
                (liste_iter, Liste_S_sc, {'label':"SingleCoil", 'color':Color_s}),
                (liste_iter, Liste_S_hb, {'label':"Humbucker", 'color':Color_h}),
                title="Convergence de l'algorithme de Levenberg-Marquardt",
                xlabel="Nombre d'iterations",
                ylabel="Signal normalise des residus [a.u.]",
                legend=True)

plt.close()
plt.plot(f, Z_sc*1e-3, '+r', markersize=ms_exp, label='Experience')
for i in range(1, 10):
    plt.plot(np.linspace(20, 20000, 1000), mod.Z_micro(Liste_Xsc[i-1], Rs, 2*np.pi*np.linspace(20, 20000, 1000))*1e-3, label=f"{i-1}")
plt.xlim(20, 20000)
plt.legend()
plt.show()

print("\n---\nFin de l'execution de Modele_elec.py\n---\n")