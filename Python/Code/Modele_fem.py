"""Modèle de la vibration d'une corde et de la tension induite dans un micro simple et un humbucker.
    """

print("\n\n---\nModele_fem.py\n---\n")
                                            #!####################!#
                                            #!   Bibliothèques    !#
                                            #!####################!#

print("\nImportation des bibliotheques...")
import numpy as np
import matplotlib.pyplot as plt
print("Bibliotheques importees.")

                                            #!####################!#
                                            #! Constante Physique !#
                                            #!####################!#

print("\nInitialisation des constantes physiques et Electromagnetiques...")
A = 1*10**(-3)          #? Amplitude initiale de vibration de la corde [m]
mu = 9*10**(-3)         #? Masse linéique de la corde [kg.m-1]
xm = 0.105              #? Distance entre la source la corde et le micro [m]
x1 = 0.140              #? Distance entre la source la corde et le micro [m]
x2 = 0.160              #? Distance entre la source la corde et le micro [m]
L = 0.84                #? Longueur de la corde [m]
x0 = 0.105              #? Origine de la perturbation

F = 72                   #? Frequence de la note [Hz]
print(f"Tension de la corde : T = {4*mu*(L*F)**2} N")
print(f"Frequence de la note : f = {F} Hz")
T = 4*mu*(L*F)**2      #? Tension de la corde [N]


                                            #!######################!#
                                            #!  Constante Elec-Mag  !#
                                            #!######################!#

N = 7000                #? Nombre de spire du micro [.]
S = 1*10**(-4)          #? Surface de la bobine du micro [m²]
dBdy = 5                #? Gradient (supposé constant) [.]
B0 = 9*10**(-3)         #? Amplitude du champ parasite [T]
fB = 50                  #? Frequence du champ parasite [s-1]
alpha = 0.02            #? ???
print("Constantes physiques et Electromagnetiques initialisees.")

                                            #!######################!#
                                            #!      Fonction        !#
                                            #!######################!#

def entrée(n: int, x: float, t: np.ndarray) -> np.ndarray:
    s = 0
    v = np.sqrt(T/mu)
    for i in range(1, n+1):
        Bn = (2*A*np.sin((i*np.pi*x0)/(L)))/((x0)/(L)*(1-(x0)/(L))*np.pi**2*i**2)
        s += Bn*np.sin(i*np.pi*x/L)*np.cos(i*np.pi*v*t/L)
    
    return s

def fem_singlecoil(n: int, N: int, xm: float, t: np.ndarray) -> np.ndarray:
    s = 0
    for i in range(1, n+1):
        fn = (i/(2*L))*np.sqrt(T/mu)
        An = 2*np.pi*fn*np.sin(i*np.pi*xm/L)
        Bn = (2*A*np.sin((i*np.pi*x0)/(L)))/((x0)/(L)*(1-(x0)/(L))*np.pi**2*i**2)
        s += An*Bn*np.sin(2*np.pi*fn*t)

    return N*S*dBdy*s

def fem_humbucker(n: int, N: int, x1: float, x2: float, t: np.ndarray) -> np.ndarray:
    s = 0
    for i in range(1, n+1):
        fn = (i/(2*L))*np.sqrt(T/mu)
        An = 2*np.pi*fn*(np.sin(i*np.pi*x1/L)+np.sin(i*np.pi*x2/L))
        Bn = (2*A*np.sin((i*np.pi*x0)/(L)))/((x0)/(L)*(1-(x0)/(L))*np.pi**2*i**2)
        s += An*Bn*np.sin(2*np.pi*fn*t)

    return N*S*dBdy*s

def B(x: float, f: float, t: np.ndarray) -> np.ndarray:
    return B0*np.sin(2*np.pi*f*t)*(1+alpha*x)

def fem_par_s(N:int, xm: float, f:float, t: np.ndarray) -> np.ndarray:
    return -N*S*B0*(1+alpha*xm)*2*np.pi*f*np.cos(2*np.pi*f*t)

def fem_par_h(N:int, x1: float, x2: float, f:float, t: np.ndarray) -> np.ndarray:
    return -N*S*B0*np.pi*f*np.cos(2*np.pi*f*t)*alpha*np.abs(x2 - x1)


                                            #!######################!#
                                            #!         Appel        !#
                                            #!######################!#

print("\nCalcul des tensions induites...")
print("Echantillonnage...")
Te = 0.05
Ne = 1000
fs = Ne/Te
Na = 20
print(f"Frequence d'echantillonnage : fs = {fs} Hz")

print("Calcul des tensions en reponse a la vibration de la corde...")
temps = np.linspace(0, Te, Ne, endpoint=False)
corde = entrée(Na, xm, temps)
fem_s = fem_singlecoil(Na, N, xm, temps)
fem_h = fem_humbucker(Na, N, x1, x2, temps)

print("Calcul des tensions en reponse au champ parasite...")
champ_par = B(xm, fB, temps)
fem_s_par = fem_par_s(N, xm, fB, temps)
fem_h_par = fem_par_h(N, x1, x2, fB, temps)

print("Calcul des tensions totales...")
fem_s_tot = fem_s + fem_s_par
fem_h_tot = fem_h + fem_h_par

                                            #!######################!#
                                            #!          FFT         !#
                                            #!######################!#


Color_c = "#505050"
Color_h = "#3344CB"
Color_s = "#CAA307"


print("\nCalcul des FFT...")
fft_s = np.fft.fft(fem_s_tot)
fft_s_amp = 2 * np.abs(fft_s) / Ne
fft_s_amp[0] /= 2

fft_h = np.fft.fft(fem_h_tot)
fft_h_amp = 2 * np.abs(fft_h) / Ne
fft_h_amp[0] /= 2

freqs = np.fft.fftfreq(Ne, 1/fs)
mask = 0 <= freqs
freqs = freqs[mask]
mask2 = freqs <= 1500
freqs = freqs[mask2]

fft_s_amp = fft_s_amp[mask]
fft_s_amp = fft_s_amp[mask2]
fft_h_amp = fft_h_amp[mask]
fft_h_amp = fft_h_amp[mask2]
print("FFT calculees.")

print("Affichage des FFT...")
Title = "Spectre de la tension en sortie des micros (Fondamental à " + str(F) + " Hz)"
plt.figure(figsize=(6,5))
plt.plot(freqs, fft_h_amp, color=Color_h, label="Micro Humbucker")
plt.plot(freqs, fft_s_amp, color=Color_s, label="Micro Simple")
plt.title(Title)
plt.legend()
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Tension [V]")
plt.grid(True)
plt.savefig("Python/Exports/FFT.png")
plt.close()
print("FFT sauvegardees sous Python/Exports/FFT.png.")

                                            #!######################!#
                                            #!   Affichage Courbes  !#
                                            #!######################!#


print("\nAffichage des courbes...")

#* --- Comparaison des micros (corde seule) ---*#
print("Comparaison des micros (corde seule)...")
fig, ax1 = plt.subplots()

# Axe 1
ax1.set_xlabel("Temps [s]")
ax1.set_ylabel("Deplacement de la corde [m]", color=Color_c)
l1, = ax1.plot(temps, corde, color=Color_c, label="Deplacement de la corde [m]")
ax1.tick_params(axis='y', labelcolor=Color_c)

# Axe 2
ax2 = ax1.twinx()  # Deuxième axe y
ax2.set_ylabel("Tension induite [V]", color="#000000")
l3, = ax2.plot(temps, fem_h, color=Color_h, label="Micro Humbucker [V]")
l2, = ax2.plot(temps, fem_s, color=Color_s, label="Micro Simple [V]")
ax2.tick_params(axis='y', labelcolor="#000000")

# Légende unique
fig.legend(handles=[l1, l3, l2], loc='lower left')

# Mise en forme
plt.title("Tension induite en sortie du micro (Vibration de la corde uniquement)")
fig.tight_layout()
plt.grid()
plt.savefig("Python/Exports/Comparaison_sh.png")
plt.close()
print("Courbes sauvegardees sous Python/Exports/Comparaison_sh.png.")

#* --- Comparaison des micros (champ parasite seul) ---*#
print("Comparaison des micros (champ parasite seul)...")
fig, ax1 = plt.subplots()

# Axe 1
ax1.set_xlabel("Temps [s]")
ax1.set_ylabel("Champ Parasite [T]", color=Color_c)
l1, = ax1.plot(temps, champ_par, color=Color_c, label="Champ Parasite [T]")
ax1.tick_params(axis='y', labelcolor=Color_c)

# Axe 2
ax2 = ax1.twinx()  # Deuxième axe y
ax2.set_ylabel("Tension induite [V]", color="#000000")
l3, = ax2.plot(temps, fem_h_par, color=Color_h, label="Micro Humbucker [V]")
l2, = ax2.plot(temps, fem_s_par, color=Color_s, label="Micro Simple [V]")
ax2.tick_params(axis='y', labelcolor="#000000")

# Légende unique
fig.legend(handles=[l1, l3, l2], loc='lower left')

# Mise en forme
plt.title("Tension induite en sortie du micro (champ parasite uniquement)")
fig.tight_layout()
plt.grid()
plt.savefig("Python/Exports/Parasite.png")
plt.close()
print("Courbes sauvegardees sous Python/Exports/Parasite.png.")

#* --- Comparaison des micros (total) ---*#
print("Comparaison des micros (total)...")

fig, ax1 = plt.subplots()

# Axe 1
ax1.set_xlabel("Temps [s]")
ax1.set_ylabel("Deplacement de la corde [m]", color=Color_c)
l1, = ax1.plot(temps, corde, color=Color_c, label="Deplacement de la corde [m]")
ax1.tick_params(axis='y', labelcolor=Color_c)

# Axe 2
ax2 = ax1.twinx()  # Deuxième axe y
ax2.set_ylabel("Tension induite [V]", color="#000000")
l3, = ax2.plot(temps, fem_h_tot, color=Color_h, label="Micro Humbucker [V]")
l2, = ax2.plot(temps, fem_s_tot, color=Color_s, label="Micro Simple [V]")
ax2.tick_params(axis='y', labelcolor="#000000")

# Légende unique
fig.legend(handles=[l1, l3, l2], loc='lower left')

# Mise en forme
plt.title("Tension induite en sortie du micro (Corde + Champ parasite)")
fig.tight_layout()
plt.grid()
plt.savefig("Python/Exports/Total.png")
plt.close()
print("Courbes sauvegardees sous Python/Exports/Total.png.")


print("\n---\nFin de l'execution de Modele_fem.py\n---\n")