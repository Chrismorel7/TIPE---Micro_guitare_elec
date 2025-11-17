"""Modèle électrique du micro de guitare electrique et affichage de la réponse en fréquence via un diagramme de Bode.
    """

print("\n\n---\nModele_elec.py\n---\n")

                                            #!####################!#
                                            #!   Bibliothèques    !#
                                            #!####################!#

print("\nImportation des bibliotheques...")
import numpy as np
import matplotlib.pyplot as plt
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

Color_c = "#505050"
Color_h = "#3344CB"
Color_s = "#CAA307"

print("Variables initialisees.")

                                            #!####################!#
                                            #! Lectures Fichiers  !#
                                            #!####################!#

print("\nLecture du fichier Bode2.txt...")

file = open("Python/data/Bode2.txt", 'r')
L_lignes = file.readlines()

for i in range(1, len(L_lignes)):
    values = L_lignes[i].strip().split("\t")
    values = [float(v.replace("\n", "")) for v in values]
    freq.append(values[0])
    Vvout_hum_db.append(values[1])
    Vvout_hum_deg.append(values[2])
    Vvout_sin_db.append(values[3])
    Vvout_sin_deg.append(values[4])
file.close()



                                            #!####################!#
                                            #!     Affichage      !#
                                            #!####################!#

print("\nAffichage du diagramme de Bode...")
plt.figure(1)

plt.subplot(2, 1, 1)
plt.semilogx(freq, Vvout_hum_db, color=Color_h, label="Gain Humbucker")
plt.semilogx(freq, Vvout_sin_db, color=Color_s, label="Gain Simple Coil")
plt.grid()
plt.ylabel("Gain (dB)")
plt.ylim(-23, 23)
plt.xlim(20, 20000)
plt.legend()
plt.title("Réponse en fréquence du micro")

plt.subplot(2, 1, 2)
plt.semilogx(freq, Vvout_hum_deg, color=Color_h, label="Phase Humbucker")
plt.semilogx(freq, Vvout_sin_deg, color=Color_s, label="Phase Simple Coil")
plt.grid()
plt.ylabel("Phase (°)")
plt.ylim(-90, 90)
plt.xlim(20, 20000)
plt.xlabel("Fréquence (Hz)")
plt.legend()

plt.savefig("Python/Exports/Bode.png")
plt.close()
print("Diagramme de Bode sauvegarde sous Python/Exports/Bode.png.")

print("\n---\nFin de l'execution de Modele_elec.py\n---\n")