"""Modele de la vibration d'une corde et de la tension induite dans un micro simple et un humbucker.
    """

print("\n\n---\nModele_fem.py\n---\n")
                                            #!####################!#
                                            #!   Bibliotheques    !#
                                            #!####################!#
print("-" * 100)
print("\nImportation des bibliotheques...")
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
print("Bibliotheques importees.")
# Configure logging for optional diagnostic output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

                                            #!###########!#
                                            #! Constante !#
                                            #!###########!#
print("-" * 100)
print("\nInitialisation des constantes")
#* Constante physique
rho = 1.68*10**(-8)
mu0 = 4e-7 * np.pi
#* Parametre maquette
xm = 0.105
x1 = 0.140
x2 = 0.160
L = 0.845
mu = 9*10**(-3)
dBdy = 0.05             #A verifier ???
#* Parametre Micros
d = 6.3*10**(-5)
S = 7*10**(-4)
#* Parametre Micro Singlecoil
RCC_s = 8.92*10**3      #A verifier ???
lmoy_s = 0.147
N_s = int(RCC_s*np.pi*d**2 / (4*rho*lmoy_s))
N_s = 8000
#* Parametre Micro Humbucker
RCC_h = 4.52*10**3
lmoy_h = lmoy_s            #A verifier ???
N_h = 2*int(RCC_h*np.pi*d**2 / (8*rho*lmoy_h)) 
#* Parametre Champ parasite
Nbob = 95
R = 0.130/2
Lbob = 1                #A verifier ???
Rbob = 10               #A verifier ???
fB = 50
U0 = 10
I0 = 7                  #A verifier ???
#* Parametre variable
A = 35*10**(-4)          #A verifier ???
x0 = 0.140
F = 68.3
T = 4*mu*(L*F)**2
#* Couleurs graphiques
Color_c = "#505050"
Color_h_test = "#3344CB"
Color_s_test = "#CAA307"
Color_h = "#7086EA"
Color_s = "#EACB07"

# Affichage des variables avec unites
print("Constantes initialisees.")
print("\nValeurs des parametres (avec unites) :")
params = [
    ("rho", rho, "Ohm.m", "Resistivite du fil de cuivre"),
    ("mu0", mu0, "H.m-1", "Permeabilite magnetique du vide"),
    ("xm", xm, "m", "Distance corde-micro"),
    ("x1", x1, "m", "Distance corde-micro humbucker 1"),
    ("x2", x2, "m", "Distance corde-micro humbucker 2"),
    ("L", L, "m", "Longueur de la corde"),
    ("mu", mu, "kg.m-1", "Masse lineique de la corde"),
    ("dBdy", dBdy, "T.m-1", "Gradient du champ magnetique"),
    ("d", d, "m", "Diametre du fil de cuivre"),
    ("S", S, "m2", "Surface de la bobine"),
    ("RCC_s", RCC_s, "Ohm", "Resistance DC micro singlecoil"),
    ("lmoy_s", lmoy_s, "m", "Longueur moyenne du fil enroule"),
    ("N_s", N_s, "spires", "Nombre de spires du micro singlecoil"),
    ("RCC_h", RCC_h, "Ohm", "Resistance DC micro humbucker"),
    ("lmoy_h", lmoy_h, "m", "Longueur moyenne du fil humbucker"),
    ("N_h", N_h, "spires", "Nombre de spires micro humbucker"),
    ("Nbob", Nbob, "spires", "Nombre de spires bobine Helmholtz"),
    ("R", R, "m", "Rayon des bobines Helmholtz"),
    ("Lbob", Lbob, "H", "Inductance bobines Helmholtz"),
    ("Rbob", Rbob, "Ohm", "Resistance bobines Helmholtz"),
    ("fB", fB, "Hz", "Frequence du champ parasite"),
    ("U0", U0, "V", "Tension bobines Helmholtz"),
    #("B0", B0, "T", "Amplitude champ magnetique parasite"),
    ("A", A, "m", "Amplitude initiale de vibration"),
    ("x0", x0, "m", "Origine de la perturbation"),
    ("F", F, "Hz", "Frequence de la note jouee"),
    ("T", T, "N", "Tension de la corde"),
]

# En-tête du tableau
print(f"{'Var':<8} {'Valeur':>18} {'Unite':<10} {'Description':<45}")
print("-" * 85)

# Avertissement si les paramètres humbucker ne sont pas renseignés
if RCC_h is None or lmoy_h is None or N_h is None:
    logging.warning("Parametres humbucker (RCC_h, lmoy_h, N_h) non defini(s). Certaines simulations pourront utiliser N_s en substitution.")


# Affichage des parametres
compteur = 1
for var, val, unit, desc in params:
    # Format de la valeur selon son type
    if isinstance(val, (int, np.integer)):
        val_str = f"{val:d}"
    elif isinstance(val, (float, np.floating)):
        val_str = f"{val:.6g}"
    else:
        val_str = str(val)
    if compteur == 1 or compteur == 3 or compteur == 9 or compteur == 11 or compteur == 14 or compteur == 17 or compteur == 24 :
        print("-"*70)
    print(f"{var:<8} {val_str:>18} {unit:<10} {desc:<45}")
    compteur += 1

print("-" * 85)

                                            #!######################!#
                                            #!      Fonction        !#
                                            #!######################!#


def entree(n: int, x: float, t: np.ndarray) -> np.ndarray:
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

def B(x:float, f: float, t: np.ndarray) -> np.ndarray:
    return (Nbob*mu0*Rbob**2)/2 * (1 / (Rbob**2 + (x-Rbob/2)**2)**(3/2) + 1 / (Rbob**2 + (x+Rbob/2)**2)**(3/2)) * I0 * np.sin(2*np.pi*f*t)

def dBdt(x:float, f: float, t: np.ndarray) -> np.ndarray:
    return (Nbob*mu0*Rbob**2)/2 * (1 / (Rbob**2 + (x-Rbob/2)**2)**(3/2) + 1 / (Rbob**2 + (x+Rbob/2)**2)**(3/2))*2*I0*np.pi*f*np.cos(2*np.pi*f*t)

def fem_par_s(N:int, xm: float, f:float, t: np.ndarray) -> np.ndarray:
    return -N*S*dBdt(0, f, t)

def fem_par_h(N:int, x1: float, x2: float, f:float, t: np.ndarray) -> np.ndarray:
    return -N*S*(dBdt(0, f, t) - dBdt(abs(x2-x1), f, t))

                                            #!######################!#
                                            #!       Resultats      !#
                                            #!######################!#

if __name__ == "__main__":
    print("Lecture des donnees de test...")

    List_Ipar = np.linspace(1, 7, 7)
    List_amp_h_th = np.zeros(7)
    List_amp_h_test = [0, 40*10**-3, 0, 0, 0, 0, 0]
    List_amp_s_th = [(N_s*S*Nbob*mu0*Rbob**2)/2 * (1 / (Rbob**2 + (0-Rbob/2)**2)**(3/2) + 1/(Rbob**2 + (0+Rbob/2)**2)**(3/2))*2*i*np.pi*fB for i in List_Ipar]
    Liste_amp_s_test = [N_h*S* (((Nbob*mu0*Rbob**2)/2 * (1 / (Rbob**2 + (0-Rbob/2)**2)**(3/2) + 1/(Rbob**2 + (0+Rbob/2)**2)**(3/2))) - ((Nbob*mu0*Rbob**2)/2 * (1 / (Rbob**2 + (abs(x2-x1)-Rbob/2)**2)**(3/2) + 1/(Rbob**2 + ((x2-x1)+Rbob/2)**2)**(3/2))))*2*i*np.pi*fB for i in List_Ipar]

    doc = open("Test/17-11-2025/Test final/Humbucker_corde.txt")
    L_lignes = doc.readlines()
    Test_temps_h = []
    Test_h = []
    for i in range(53195, 93195):
        values = L_lignes[i].strip().split("\t")
        values = [float(v.replace("\n", "")) for v in values]
        Test_temps_h.append(values[0])
        Test_h.append(values[1])

    Test_temps_h = np.array(Test_temps_h)
    Test_temps_h -= Test_temps_h[0]
    Test_temps_h_fft = np.array(Test_temps_h)
    Test_temps_h = np.array(Test_temps_h[:4000])
    Test_h_fft = np.array(Test_h)
    Test_h = np.array(Test_h[:4000])
    doc.close()
    print("Donnees de test Humbucker corde seule lues.")

    doc = open("Test/17-11-2025/Test final/Humbucker Parasite seul.txt")
    L_lignes = doc.readlines()
    Test_temps_h_p = []
    Test_h_p = []
    for i in range(1, 4001):
        values = L_lignes[i].strip().split("\t")
        values = [float(v.replace("\n", "")) for v in values]
        Test_temps_h_p.append(values[0])
        Test_h_p.append(values[1])
    
    Test_temps_h_p = np.array(Test_temps_h_p)
    Test_temps_h_p -= Test_temps_h_p[0]
    Test_h_p = np.array(Test_h_p)
    doc.close()
    print("Donnees de test Humbucker parasite seules lues.")
    
    
    doc= open("Test/17-11-2025/Test final/Humbucker parasite + corde.txt")
    L_lignes = doc.readlines()
    Test_temps_tot = []
    Test_h_tot = []
    for i in range(34360, 74360):
        values = L_lignes[i].strip().split("\t")
        values = [float(v.replace("\n", "")) for v in values]
        Test_temps_tot.append(values[0])
        Test_h_tot.append(values[1])

    Test_temps_h_tot = np.array(Test_temps_tot)
    Test_temps_h_tot -= Test_temps_h_tot[0]
    Test_temps_h_tot_fft = np.array(Test_temps_h_tot)
    Test_temps_h_tot = np.array(Test_temps_h_tot[:4000])
    Test_h_tot_fft = np.array(Test_h_tot)
    Test_h_tot = np.array(Test_h_tot[:4000])
    doc.close()
    print("Donnees de test Humbucker parasite + corde lues.")
    
    

print("-" * 100)
                                            #!######################!#
                                            #!         Appel        !#
                                            #!######################!#
if __name__ == "__main__":
    # Ensure export directory exists
    os.makedirs("Python/Exports", exist_ok=True)

    print("-" * 100)
    print("\nCalcul des tensions induites...")
    print("\nEchantillonnage...")
    Te = 4000*0.00001
    Ne = 4000
    fs = Ne/Te
    Na = 20
    
    Te_fft = 1
    Ne_fft = 100_000
    fs_fft = Ne_fft/Te_fft
    
    # Affichage des parametres d'echantillonnage
    print("\nParametres d'echantillonnage (avec unites) :")
    print(f"Te = {Te} s")
    print(f"Ne = {Ne} echantillons")
    print(f"fs = {fs:.6g} Hz")
    print(f"Na = {Na}")

    print("\nCalcul des tensions en reponse a la vibration de la corde...")
    temps = np.linspace(0, Te, Ne, endpoint=False)
    corde = entree(Na, xm, temps)
    fem_s = fem_singlecoil(Na, N_s, xm, temps)
    fem_h = fem_humbucker(Na, N_h, x1, x2, temps)

    print("\nCalcul des tensions en reponse au champ parasite...")
    champ_par = B(0, fB, temps)
    fem_s_par = fem_par_s(N_s, xm, fB, temps)
    fem_h_par = fem_par_h(N_h, x1, x2, fB, temps)

    print("\nCalcul des tensions totales...")
    fem_s_tot = fem_s + fem_s_par
    fem_h_tot = fem_h + fem_h_par

    # Option pour tracer la FFT (ne s'exécute que si True)
    tracer_fft = True
    if tracer_fft:
        print("-" * 100)
        print("\nCalcul des FFT...")
        print("\nParametres d'echantillonnage (avec unites) :")
        print(f"Te = {Te_fft} s")
        print(f"Ne = {Ne_fft} echantillons")
        print(f"fs = {fs_fft:.6g} Hz")
        print(f"Na = {Na}")
        
        print("\nCalcul des tensions en reponse a la vibration de la corde...")
        temps_fft = Test_temps_h_fft #np.linspace(0, Te_fft, Ne_fft, endpoint=False)
        corde_fft = entree(Na, xm, Test_temps_h_fft)
        fem_s_fft = fem_singlecoil(Na, N_s, xm, Test_temps_h_fft)
        fem_h_fft = fem_humbucker(Na, N_s, x1, x2, Test_temps_h_fft)
        
        print("\nCalcul des tensions en reponse au champ parasite...")
        champ_par_fft = B(0, fB, temps_fft)
        fem_s_par_fft = fem_par_s(N_s, xm, fB, temps_fft)
        fem_h_par_fft = fem_par_h(N_s, x1, x2, fB, temps_fft)
        
        print("\nCalcul des tensions totales...")
        fem_s_tot_fft = fem_s_fft + fem_s_par_fft
        fem_h_tot_fft = fem_h_fft + fem_h_par_fft
        
        fft_s = np.fft.fft(fem_s_tot_fft)
        fft_s_amp = 2 * np.abs(fft_s) / Ne_fft
        fft_s_amp[0] /= 2

        fft_h = np.fft.fft(fem_h_tot_fft)
        fft_h_amp = 2 * np.abs(fft_h) / Ne_fft
        fft_h_amp[0] /= 2
        
        fft_h_test = np.fft.fft(Test_h_tot_fft)
        fft_h_test_amp = 2 * np.abs(fft_h_test) / Ne_fft
        fft_h_test_amp[0] /= 2

        freqs = np.fft.fftfreq(40000, 1/fs_fft)
        mask = 0 <= freqs
        freqs = freqs[mask]
        mask2 = freqs <= 800
        freqs = freqs[mask2]

        fft_s_amp = fft_s_amp[mask]
        fft_s_amp = fft_s_amp[mask2]
        fft_h_amp = fft_h_amp[mask]
        fft_h_amp = fft_h_amp[mask2]
        fft_h_test_amp = fft_h_test_amp[mask]
        fft_h_test_amp = fft_h_test_amp[mask2]
        
        
        max_amp_h, max_amp_s, max_amp_test_h = np.max(fft_h_amp), np.max(fft_s_amp), np.max(fft_h_test_amp)
        fft_h_amp = fft_h_amp / max_amp_h
        fft_s_amp = fft_s_amp / max_amp_s
        fft_h_test_amp = fft_h_test_amp / max_amp_test_h
        
        print("FFT calculees.")

        print("Affichage des FFT...")
        Title = "Spectre de la tension en sortie des micros (Fondamental à " + str(F) + " Hz)"
        plt.figure(figsize=(6,5))
        plt.plot(freqs, fft_h_amp, color=Color_h, label="Micro Humbucker - Theorique")
        plt.plot(freqs, fft_h_test_amp, "+", color=Color_h_test, label="Micro Humbucker - Test")
        plt.title(Title)
        plt.legend()
        plt.xlabel("Frequence (Hz)")
        plt.ylabel("Tension normalisee")
        plt.grid(True)
        plt.savefig("Python/Exports/00-FFT_h.png")
        plt.close()
        print("FFT sauvegardees sous Python/Exports/00-FFT_h.png.")
        
        plt.figure(figsize=(6,5))
        plt.plot(freqs, fft_s_amp, color=Color_s, label="Micro Singlecoil - Theorique")
        #plt.plot(freqs, fft_h_test_amp, "+", color=Color_h_test, label="Micro Humbucker - Test")
        plt.title(Title)
        plt.legend()
        plt.xlabel("Frequence (Hz)")
        plt.ylabel("Tension normalisee")
        plt.grid(True)
        plt.savefig("Python/Exports/00-FFT_s.png")
        plt.close()
        print("FFT sauvegardees sous Python/Exports/00-FFT_s.png.")

        plt.figure(figsize=(6,5))
        plt.plot(freqs, fft_h_amp, color=Color_h, label="Micro Humbucker - Théorique")
        plt.plot(freqs, fft_s_amp, color=Color_s, label="Micro Singlecoil - Theorique")
        plt.title(Title)
        plt.legend()
        plt.xlabel("Frequence (Hz)")
        plt.ylabel("Tension normalisee")
        plt.grid(True)
        plt.savefig("Python/Exports/00-FFT_th.png")
        plt.close()
        print("FFT sauvegardees sous Python/Exports/00-FFT_th.png.")
        
    print("-" * 100)
    print("\nAffichage des courbes...")

    '''
    #* --- Micro Singlecoil (corde) ---*#
    print("Micro Singlecoil seul...")
    plt.plot(Test_temps_s, fem_s, color=Color_s, label="Micro Simple [V]")
    plt.plot(Test_temps_s, Test_s, "+", color=Color_s, label="Donnees de test [V]")
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    plt.title("Tension induite en sortie du micro Simple - Corde seule")
    plt.xlabel("Temps [s]")
    plt.ylabel("Tension [V]")
    plt.gca().xaxis.set_major_locator(MultipleLocator(10e-3))
    plt.gca().yaxis.set_major_locator(MultipleLocator(100e-3))
    plt.grid(which='major', color='black', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.savefig("Python/Exports/02-Simplecoil_corde.png")
    plt.close()
    print("Courbes sauvegardees sous Python/Exports/02-Simplecoil_corde.png.")
    '''
    #* --- Micro Humbucker (corde) ---*#
    print("Micro Humbucker (corde)...")
    plt.plot(temps, Test_h, "+", color=Color_h_test, label="Micro Humbucker - Test [V]")
    plt.plot(temps, fem_h, color=Color_h, label="Micro Humbucker - Theorique [V]")
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    plt.title("Tension induite en sortie du micro Humbucker - Corde seule")
    plt.xlabel("Temps [s]")
    plt.ylabel("Tension [V]")
    plt.grid()
    plt.legend()
    plt.savefig("Python/Exports/01-Humbucker_corde.png")
    plt.close()
    print("Courbes sauvegardees sous Python/Exports/01-Humbucker_corde.png.")
    
    #* --- Micro Humbucker (Parasite) ---*#
    print("Micro Humbucker (Parasite)...")
    plt.plot(temps, Test_h_p, "+", color=Color_h_test, label="Micro Humbucker - Test [V]")
    plt.plot(temps, fem_h_par, color=Color_h, label="Micro Humbucker - Theorique [V]")
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    plt.title("Tension induite en sortie des micros - Parasite seul")
    plt.xlabel("Temps [s]")
    plt.ylabel("Tension [V]")
    plt.ylim(-0.5, 0.5)
    plt.grid()
    plt.legend()
    plt.savefig("Python/Exports/02-Humbucker_parasite.png")
    plt.close()
    print("Courbes sauvegardees sous Python/Exports/02-Humbucker_parasite.png.")

    
    #* --- Humbucker (total) ---*#
    print("Micro Humbucker (total)...")
    plt.plot(temps, Test_h_tot, "+", color=Color_h_test, label="Micro Humbucker - Test [V]")
    plt.plot(temps, fem_h_tot, color=Color_h, label="Micro Humbucker - Theorique [V]")
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    plt.title("Tension induite en sortie du micro Humbucker - Total")
    plt.xlabel("Temps [s]")
    plt.ylabel("Tension [V]")
    plt.grid()
    plt.legend()
    plt.savefig("Python/Exports/03-Humbucker_total.png")
    plt.close()
    print("Courbes sauvegardees sous Python/Exports/03-Humbucker_total.png.")

    
    #* --- Total ---*#
    print("Tous micros total...")
    plt.plot(temps, Test_h_tot, "+", color=Color_h_test, label="Micro Humbucker - Test [V]")
    plt.plot(temps, fem_h_tot, color=Color_h, label="Micro Humbucker - Theorique [V]")
    plt.plot(temps, fem_s_tot, color=Color_s, label="Micro Singlecoil - Theorique [V]")
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    plt.title("Tension induite en sortie des micros")
    plt.xlabel("Temps [s]")
    plt.ylabel("Tension [V]")
    plt.grid()
    plt.legend()
    plt.savefig("Python/Exports/99-total.png")
    plt.close()
    print("Courbes sauvegardees sous Python/Exports/99-total.png.")
    
    #* --- Comparaison des micros Theorique (corde seule) ---*#
    print("Comparaison des micros  Theorique (corde seule)...")
    fig, ax1 = plt.subplots()
    ax1.set_ylabel("Tension induite [V]", color="#000000")
    l3, = ax1.plot(temps, fem_h, color=Color_h, label="Micro Humbucker - Theorique [V]")
    l2, = ax1.plot(temps, fem_s, color=Color_s, label="Micro Singlecoil - Theorique [V]")
    ax1.tick_params(axis='y', labelcolor="#000000")
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    fig.legend(handles=[l3, l2], loc='lower left')
    plt.title("Tension induite en sortie des micros - Corde seule")
    fig.tight_layout()
    plt.grid()
    plt.savefig("Python/Exports/Th-Corde.png")
    plt.close()
    print("Courbes sauvegardees sous Python/Exports/Th-Corde.png.")
    
    #* --- Comparaison des micros Theorique (total) ---*#
    print("Comparaison des micros  Theorique (total)...")
    plt.plot(temps, fem_h_tot, color=Color_h, label="Micro Humbucker - Theorique [V]")
    plt.plot(temps, fem_s_tot, color=Color_s, label="Micro Singlecoil - Theorique [V]")
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    plt.title("Tension induite en sortie des micros")
    plt.xlabel("Temps [s]")
    plt.ylabel("Tension [V]")
    plt.grid()
    plt.legend()
    plt.savefig("Python/Exports/Th-Total.png")
    plt.close()
    print("Courbes sauvegardees sous Python/Exports/Th-Total.png.")

    #* --- Droite champ parasite ---*#
    print("Comparaison des micros champ parasite...")
    plt.plot(List_Ipar, List_amp_h_th, color=Color_h, label="Micro Humbucker - Theorique [V]")
    plt.plot(List_Ipar, List_amp_h_test, "+", color=Color_h_test, label="Micro Humbucker - Test [V]")
    plt.plot(List_Ipar, List_amp_s_th, color=Color_s, label="Micro Singlecoil - Theorique [V]")
    plt.plot(List_Ipar, Liste_amp_s_test, "+", color=Color_s_test, label="Micro Singlecoil - Test [V]")
    #plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(-6, -6))
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    plt.title("Amplitude des tension induite en sortie des micros soumis à un champ parasite")
    plt.xlabel("Amplitude du champ parasite [A]")
    plt.ylabel("Ampltitude Tension [V]")
    plt.grid()
    plt.legend()
    plt.savefig("Python/Exports/99-Doite.png")
    plt.close()
    print("Courbes sauvegardees sous Python/Exports/99-Droite.png.")

print("-" * 100)
print("\n---\nFin de l'execution de Modele_fem.py\n---\n")