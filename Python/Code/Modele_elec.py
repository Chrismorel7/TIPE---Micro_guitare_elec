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

f0_sc_exp = 5800   # Hz
Q_sc_exp  = 1.5    # Adimensionnel

f0_hb_exp = 7800   # Hz
Q_hb_exp  = 1.3    # Adimensionnel

# --- 2. Fonction de Transfert (Modèle Passe-Bas 2nd Ordre) ---

def tracer_bode_identification(Rs, Ramp, freq, Z_meas):
    # 1. Identification des paramètres parasites
    
    # Fréquence de résonance f0 (là où l'impédance est maximale)
    idx_peak = np.argmax(Z_meas)
    f0 = freq[idx_peak]
    
    # Capacité parasite Cs
    # On utilise un point à très haute fréquence où le comportement est capacitif (Z ~ 1/Cw)
    # On prend le dernier point du tableau (fréquence max)
    f_hf = freq[-1]
    Z_hf = Z_meas[-1]
    Cs_est = 1 / (2 * np.pi * f_hf * Z_hf)
    
    # Inductance Ls
    # A la résonance : LCw^2 = 1  => L = 1 / (C * (2*pi*f0)^2)
    w0 = 2 * np.pi * f0
    Ls_est = 1 / (Cs_est * w0**2)    
    # 2. Calcul de la Fonction de Transfert H(jw) du filtre complet
    # Filtre : Diviseur de tension avec Z_micro_serie (Rs + Ls) et Z_charge (Cs // Ramp)
    
    w = 2 * np.pi * freq
    
    # Impédances complexes
    Z_serie = Rs + 1j * w * Ls_est             # Branche série : R + L
    Z_parallele = Ramp / (1 + 1j * w * Ramp * Cs_est) # Branche parallèle : R_amp // Cs
    
    # Fonction de transfert H = Vout / Vin = Z_parallele / (Z_serie + Z_parallele)
    H = Z_parallele / (Z_serie + Z_parallele)
    
    # Gain en dB
    Gain_dB = 20 * np.log10(np.abs(H))
    
    return Gain_dB, f0, Ls_est, Cs_est

def phase_passe_bas_2nd_ordre(f, f0, Q):
    """
    Calcule la phase en degrés d'un filtre passe-bas du 2nd ordre.
    """
    w_w0 = f / f0
    # On utilise angle() sur le nombre complexe pour gérer correctement les quadrants (notamment après la résonance où la phase passe à -180°)
    denominateur = 1 - w_w0**2 + 1j * (w_w0 / Q)
    H = 1 / denominateur
    return np.degrees(np.angle(H))

def lire_fichier(PATH):
    f = []
    Vin = []
    Vs = []
    Vh = []
    file_s2 = open(PATH)
    L_lignes = file_s2.readlines()

    print(L_lignes)
    
    for ligne in L_lignes[1:]:
        ligne = ligne.strip().split(',')
        print(ligne)
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


Vin, f, Vs, Vh = lire_fichier("Python/raw data/Bodeexp.csv")

Z_sc = Ramp * (Vs / (24000 - Vs))
Z_hb = Ramp * (Vh / (24000 - Vh))


print("\nLecture du fichier Bode.csv...")

file = open("Python/raw data/Bode.csv", 'r')
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


gain_sc, f0_sc_exp, Lsc_exp, Csc_exp = tracer_bode_identification(5.3e3, 1e6, f, Z_sc)
gain_hb, f0_hb_exp, Lhb_exp, Chb_exp = tracer_bode_identification(4.52e3, 1e6, f, Z_hb)



phase_sc = phase_passe_bas_2nd_ordre(np.array(freq), f0_sc_exp, Q_sc_exp)
phase_hb = phase_passe_bas_2nd_ordre(np.array(freq), f0_hb_exp, Q_hb_exp)

                                            #!####################!#
                                            #!     Affichage      !#
                                            #!####################!#

print("\nAffichage du diagramme de Bode...")
plt.figure(1)

plt.subplot(2, 1, 1)
plt.semilogx(freq, Vvout_hum_db, color=Color_h, label="Humbucker (Simule)")
plt.semilogx(freq, Vvout_sin_db, color=Color_s, label="SingleCoil (Simule)")
plt.semilogx(f, gain_sc, color=Color_s_test, linewidth=2, linestyle='--', label=f'Single Coil experimental')
plt.semilogx(f, gain_hb, color=Color_h_test, linewidth=2, linestyle='--',label=f'Humbucker experimental')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylabel("Gain (dB)")
plt.ylim(-30, 30)
plt.xlim(20, 20000)
plt.title("Réponse en fréquence du micro")

plt.subplot(2, 1, 2)
plt.semilogx(freq, Vvout_hum_deg, color=Color_h, label="Humbucker (Simule)")
plt.semilogx(freq, Vvout_sin_deg, color=Color_s, label="SingleCoil (Simule)")
plt.semilogx(freq, phase_sc, color=Color_s_test, linewidth=2, linestyle='--', label=f'Single Coil experimental')
plt.semilogx(freq, phase_hb, color=Color_h_test, linewidth=2, linestyle='--',label=f'Humbucker experimental')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylabel("Phase (°)")
plt.ylim(-180, 45)
plt.xlim(20, 20000)
plt.legend(loc='lower left')
plt.xlabel("Fréquence (Hz)")

plt.savefig("Python/Exports/Freq/00-Bode.png")
plt.close()
print("Diagramme de Bode sauvegarde sous Python/Exports/00-Bode.png.")

plt.figure(figsize=(6,5))
plt.semilogx(f, Z_sc/1000, marker='+', markersize=10, linestyle=':', color=Color_s_test, label="Micro SingleCoil")
plt.semilogx(f, Z_hb/1000, marker='+', markersize=10, linestyle=':', color=Color_h_test, label="Micro Humbucker")
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylabel("Impedance [$\Omega$]")
plt.legend(loc='upper left')
plt.title("Impedances des micros en fonction de la fréquence")
plt.xlabel("Fréquence [$Hz$]")
plt.savefig("Python/Exports/Freq/Impedance.png")

print(f"Frequence de coupure theorique Single Coil : {1/(2*np.pi*np.sqrt(Ls*Cs))} Hz")
print(f"Données experimentales SC :\tf0 = {f0_sc_exp} Hz\tLs = {Lsc_exp} H\tCs = {Csc_exp} F")
print("")
print(f"Frequence de coupure theorique Humbucker : {1/(2*np.pi*np.sqrt(Lh*Ch))} Hz")
print(f"Données experimentales HB :\tf0 = {f0_hb_exp} Hz\tLh = {Lhb_exp} H\tCh = {Chb_exp} F")



print("\n---\nFin de l'execution de Modele_elec.py\n---\n")