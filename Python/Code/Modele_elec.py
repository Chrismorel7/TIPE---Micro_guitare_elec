"""Modèle électrique du micro de guitare electrique et affichage de la réponse en fréquence via un diagramme de Bode.
    """

print("\n\n---\nModele_elec.py\n---\n")

                                            #!####################!#
                                            #!   Bibliothèques    !#
                                            #!####################!#

print("\nImportation des bibliotheques...")
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
print("Bibliotheques importees.")

                                            #!####################!#
                                            #!     Variables      !#
                                            #!####################!#

print("\nInitialisation des variables...")
freq = []
Vvout_hum_db = []
Vvout_hum_deg = []
Vvout_sin_db = []
Vvout_sin_deg = []

Rs = 5.3e3          # Résistance du single coil en ohms
Ls = 2.5        # Inductance du single coil en henrys
Cs = 150e-12       # Capacité parasite du single coil en far

Rh = 4.52e3        # Résistance du humbucker en ohms
Lh = 5       # Inductance du humbucker en henrys
Ch = 250e-12      # Capacité parasite du humbucker en farads

Ramp = 1e6         # Résistance de l'ampli en ohms

Color_c = "#505050"
Color_h_test = "#4C9E31"
Color_s_test = "#C63032"
Color_h = "#3E77B2"
Color_s = "#EF812C"

print("Variables initialisees.")

                                            #!####################!#
                                            #! Lectures Fichiers  !#
                                            #!####################!#

print("\nLecture du fichier Bode.csv...")

file = open("Python/data/Bode.csv", 'r')
L_lignes = file.readlines()

for i in range(1, len(L_lignes)):
    values = L_lignes[i].strip().split("\t")
    values = [float(v.replace("\n", "")) for v in values]
    freq.append(values[0])
    Vvout_sin_db.append(values[3])
    Vvout_sin_deg.append(values[4])
    Vvout_hum_db.append(values[1])
    Vvout_hum_deg.append(values[2])
file.close()

                                            #!####################!#
                                            #!     Affichage      !#
                                            #!####################!#

print("\nAffichage du diagramme de Bode...")
plt.figure(1)

plt.subplot(2, 1, 1)
plt.semilogx(freq, Vvout_hum_db, color=Color_h, label="Humbucker (Simule)")
plt.semilogx(freq, Vvout_sin_db, color=Color_s, label="SingleCoil (Simule)")
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylabel("Gain (dB)")
plt.ylim(-30, 30)
plt.xlim(20, 20000)
plt.legend(loc='lower left')
plt.title("Réponse en fréquence du micro")

plt.subplot(2, 1, 2)
plt.semilogx(freq, Vvout_hum_deg, color=Color_h, label="Humbucker (Simule)")
plt.semilogx(freq, Vvout_sin_deg, color=Color_s, label="SingleCoil (Simule)")
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylabel("Phase (°)")
plt.ylim(-180, 45)
plt.xlim(20, 20000)
plt.xlabel("Fréquence (Hz)")

plt.savefig("Python/Exports/00-Bode.png")
plt.close()
print("Diagramme de Bode sauvegarde sous Python/Exports/00-Bode.png.")

print(f"Frequence de coupure theorique Single Coil : {1/(2*np.pi*np.sqrt(Ls*Cs))} Hz")
print(f"Frequence de coupure theorique Humbucker : {1/(2*np.pi*np.sqrt(Lh*Ch))} Hz")

print("\n---\nFin de l'execution de Modele_elec.py\n---\n")