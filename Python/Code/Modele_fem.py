"""Modele de la vibration d'une corde et de la tension induite dans un micro simple
et un humbucker.
Auteur : Christophe Morel
Date : Novembre 2025
"""

print("\n\n---\nModele_fem.py\n---\n")

#! ------------------ Bibliotheques ------------------------------
print("-" * 100)
print("\nImportation des bibliotheques...")
import os, logging, numpy as np, matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
print("Bibliotheques importees.")

#! ------------------ Constantes ---------------------------------
print("-" * 100)
print("\nInitialisation des constantes")
#* Constante physique
rho = 1.68 * 10**(-8)
mu0 = 4e-7 * np.pi
#* Parametre maquette
xm = 0.105
x1 = 0.140
x2 = 0.160
L = 0.845
mu = 9 * 10**(-3)
dBdy = 0.05             #A verifier ???
#* Parametre Micros
d = 6.3 * 10**(-5)
S = 7 * 10**(-4)
#* Parametre Micro Singlecoil
RCC_s = 5.3 * 10**3
lmoy_s = 0.147
N_s = int(RCC_s * np.pi * d**2 / (4 * rho * lmoy_s))
#* Parametre Micro Humbucker
RCC_h = 4.52 * 10**3
lmoy_h = lmoy_s/2            #A verifier ???
N_h = int(RCC_h * np.pi * d**2 / (8 * rho * lmoy_h))
#* Parametre Champ parasite
Nbob = 95
R = 0.130 / 2
Lbob = 1                #A verifier ???
Rbob = 10               #A verifier ???
fB = 50
U0 = 10
I0 = 7                  #A verifier ???
#* Parametre variable
A = 35 * 10**(-4)          #A verifier ???
x0 = 0.140                  #A verifier ???
F = 68.3                    #A verifier ???
T = 4 * mu * (L * F)**2
#* Couleurs graphiques
Color_c = "#505050"
Color_h_test = "#3344CB"
Color_s_test = "#CAA307"
Color_h = "#7086EA"
Color_s = "#EACB07"

if RCC_h is None or lmoy_h is None or N_h is None:
    logging.warning("Parametres humbucker (RCC_h, lmoy_h, N_h) non defini(s). Certaines simulations pourront utiliser N_s en substitution.")

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
    ("A", A, "m", "Amplitude initiale de vibration"),
    ("x0", x0, "m", "Origine de la perturbation"),
    ("F", F, "Hz", "Frequence de la note jouee"),
    ("T", T, "N", "Tension de la corde"),
]

try :
    print(f"{'Var':<8} {'Valeur':>18} {'Unite':<10} {'Description':<45}")
    print("-" * 85)
    compteur = 1
    for var, val, unit, desc in params:
        if isinstance(val, (int, np.integer)):
            val_str = f"{val:d}"
        elif isinstance(val, (float, np.floating)):
            val_str = f"{val:.6g}"
        else:
            val_str = str(val)
        if compteur in (1, 3, 9, 11, 14, 17, 24):
            print("-" * 70)
        print(f"{var:<8} {val_str:>18} {unit:<10} {desc:<45}")
        compteur += 1
except Exception as e:
    logging.error(f"Erreur lors de l'affichage des parametres : {e}")

print("-" * 85)


# ------------------ Fonctions numeriques ------------------------
def entree(n: int, x: float, t: np.ndarray) -> np.ndarray:
    s = 0
    v = np.sqrt(T / mu)
    for i in range(1, n + 1):
        Bn = (2 * A * np.sin((i * np.pi * x0) / (L))) / ((x0) / (L) * (1 - (x0) / (L)) * np.pi**2 * i**2)
        s += Bn * np.sin(i * np.pi * x / L) * np.cos(i * np.pi * v * t / L)
    return s


def fem_singlecoil(n: int, N: int, xm: float, t: np.ndarray) -> np.ndarray:
    s = 0
    for i in range(1, n + 1):
        fn = (i / (2 * L)) * np.sqrt(T / mu)
        An = 2 * np.pi * fn * np.sin(i * np.pi * xm / L)
        Bn = (2 * A * np.sin((i * np.pi * x0) / (L))) / ((x0) / (L) * (1 - (x0) / (L)) * np.pi**2 * i**2)
        s += An * Bn * np.sin(2 * np.pi * fn * t)
    return N * S * dBdy * s


def fem_humbucker(n: int, N: int, x1: float, x2: float, t: np.ndarray) -> np.ndarray:
    s = 0
    for i in range(1, n + 1):
        fn = (i / (2 * L)) * np.sqrt(T / mu)
        An = 2 * np.pi * fn * (np.sin(i * np.pi * x1 / L) + np.sin(i * np.pi * x2 / L))
        Bn = (2 * A * np.sin((i * np.pi * x0) / (L))) / ((x0) / (L) * (1 - (x0) / (L)) * np.pi**2 * i**2)
        s += An * Bn * np.sin(2 * np.pi * fn * t)
    return N * S * dBdy * s


def B(x: float, f: float, t: np.ndarray) -> np.ndarray:
    return (Nbob * mu0 * Rbob**2) / 2 * (1 / (Rbob**2 + (x - Rbob / 2)**2)**(3 / 2) + 1 / (Rbob**2 + (x + Rbob / 2)**2)**(3 / 2)) * I0 * np.sin(2 * np.pi * f * t)


def dBdt(x: float, f: float, t: np.ndarray) -> np.ndarray:
    return (Nbob * mu0 * Rbob**2) / 2 * (1 / (Rbob**2 + (x - Rbob / 2)**2)**(3 / 2) + 1 / (Rbob**2 + (x + Rbob / 2)**2)**(3 / 2)) * 2 * I0 * np.pi * f * np.cos(2 * np.pi * f * t)


def fem_par_s(N: int, xm: float, f: float, t: np.ndarray) -> np.ndarray:
    return -N * S * dBdt(0, f, t)


def fem_par_h(N: int, x1: float, x2: float, f: float, t: np.ndarray) -> np.ndarray:
    return -N * S * (dBdt(0, f, t) - dBdt(abs(x2 - x1), f, t))


# ------------------ Utilitaires (lecture / plot / fft) ----------
def read_test_file(path: str, line_start: int, line_end: int):
    """Lit les lignes [line_start, line_end) d'un fichier tabulé et renvoie deux tableaux temps,valeur."""
    with open(path, 'r') as doc:
        L_lignes = doc.readlines()
    times = []
    vals = []
    for i in range(line_start, line_end):
        values = L_lignes[i].strip().split("\t")
        values = [float(v.replace("\n", "")) for v in values]
        times.append(values[0])
        vals.append(values[1])
    times = np.array(times)
    times -= times[0]
    return times, np.array(vals)


def read_test_file_partial(path: str, line_start: int, line_end: int, keep_all=False):
    """Version qui lit et n'enlève pas la ligne 0 si besoin (utilisé pour petits fichiers)."""
    with open(path, 'r') as doc:
        L_lignes = doc.readlines()
    times = []
    vals = []
    for i in range(line_start, line_end):
        values = L_lignes[i].strip().split("\t")
        values = [float(v.replace("\n", "")) for v in values]
        times.append(values[0])
        vals.append(values[1])
    times = np.array(times)
    times -= times[0]
    return times, np.array(vals)


def save_plot(fig_path: str, *plot_calls, title=None, xlabel=None, ylabel=None, ylim=None, legend=True):
    """Utility to plot provided plot calls (each is a tuple: (x,y,kwargs_or_label))"""
    plt.figure(figsize=(6, 5))
    for call in plot_calls:
        x, y, kwargs = call
        plt.plot(x, y, **kwargs)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid()
    if legend:
        plt.legend()
    plt.savefig(fig_path)
    plt.close()

def normalize_signal(signal: np.ndarray):
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val

def compute_and_normalize_fft(signal: np.ndarray, Ne_fft: int):
    fft = np.fft.fft(signal)
    amp = 2 * np.abs(fft) / Ne_fft
    amp[0] /= 2
    return amp


# ------------------ Resultats / Execution ----------------------
if __name__ == "__main__":
    print("Lecture des donnees de test...")

    List_Ipar = np.linspace(1, 7, 7)
    List_amp_h_th = np.zeros(7)
    List_amp_h_test = [0, 40 * 10**-3, 0, 0, 0, 0, 0]
    List_amp_s_th = [(N_s * S * Nbob * mu0 * Rbob**2) / 2 * (1 / (Rbob**2 + (0 - Rbob / 2)**2)**(3 / 2) + 1 / (Rbob**2 + (0 + Rbob / 2)**2)**(3 / 2)) * 2 * i * np.pi * fB for i in List_Ipar]
    Liste_amp_s_test = [N_h * S * (((Nbob * mu0 * Rbob**2) / 2 * (1 / (Rbob**2 + (0 - Rbob / 2)**2)**(3 / 2) + 1 / (Rbob**2 + (0 + Rbob / 2)**2)**(3 / 2))) - ((Nbob * mu0 * Rbob**2) / 2 * (1 / (Rbob**2 + (abs(x2 - x1) - Rbob / 2)**2)**(3 / 2) + 1 / (Rbob**2 + ((x2 - x1) + Rbob / 2)**2)**(3 / 2)))) * 2 * i * np.pi * fB for i in List_Ipar]

    # Lire fichiers tests exactement comme dans la version originale
    doc_path = "Test/17-11-2025/Test final/Humbucker_corde.txt"
    try:
        with open(doc_path) as doc:
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
        print("Donnees de test Humbucker corde seule lues.")
    except Exception as e:
        logging.error(f"Erreur lors de la lecture des donnees de test Humbucker corde seule : {e}")

    doc_path = "Test/17-11-2025/Test final/Humbucker Parasite seul.txt"
    try:
        with open(doc_path) as doc:
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
        print("Donnees de test Humbucker parasite seules lues.")
    except Exception as e:
        logging.error(f"Erreur lors de la lecture des donnees de test Humbucker parasite seules : {e}")

    doc_path = "Test/17-11-2025/Test final/Humbucker parasite + corde.txt"
    try:
        with open(doc_path) as doc:
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
        print("Donnees de test Humbucker parasite + corde lues.")
    except Exception as e:
        logging.error(f"Erreur lors de la lecture des donnees de test Humbucker parasite + corde : {e}")
    print("-" * 100)

    # Ensure export directory exists
    os.makedirs("Python/Exports", exist_ok=True)

    print("-" * 100)
    print("\nCalcul des tensions induites...")
    print("\nEchantillonnage...")

    Na = 20
    Te = 4000 * 0.00001
    Ne = 4000
    fs = Ne / Te
    Te_fft = 1
    Ne_fft = 100_000
    fs_fft = Ne_fft / Te_fft

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
        temps_fft = Test_temps_h_fft
        corde_fft = entree(Na, xm, Test_temps_h_fft)
        fem_s_fft = fem_singlecoil(Na, N_s, xm, Test_temps_h_fft)
        fem_h_fft = fem_humbucker(Na, N_h, x1, x2, Test_temps_h_fft)

        print("\nCalcul des tensions en reponse au champ parasite...")
        champ_par_fft = B(0, fB, temps_fft)
        fem_s_par_fft = fem_par_s(N_s, xm, fB, temps_fft)
        fem_h_par_fft = fem_par_h(N_s, x1, x2, fB, temps_fft)

        print("\nCalcul des tensions totales...")
        fem_s_tot_fft = fem_s_fft + fem_s_par_fft
        fem_h_tot_fft = fem_h_fft + fem_h_par_fft

        fft_s_amp = compute_and_normalize_fft(fem_s_tot_fft, Ne_fft)
        fft_h_amp = compute_and_normalize_fft(fem_h_tot_fft, Ne_fft)
        fft_h_test_amp = compute_and_normalize_fft(Test_h_tot_fft, Ne_fft)

        freqs = np.fft.fftfreq(40000, 1 / fs_fft)
        mask = 0 <= freqs
        freqs = freqs[mask]
        mask2 = freqs <= 800
        freqs = freqs[mask2]

        fft_s_amp = fft_s_amp[mask][mask2]
        fft_h_amp = fft_h_amp[mask][mask2]
        fft_h_test_amp = fft_h_test_amp[mask][mask2]

        max_amp_h, max_amp_s, max_amp_test_h = np.max(fft_h_amp), np.max(fft_s_amp), np.max(fft_h_test_amp)
        fft_h_amp = fft_h_amp / max_amp_h
        fft_s_amp = fft_s_amp / max_amp_s
        fft_h_test_amp = fft_h_test_amp / max_amp_test_h

        print("FFT calculees.")

        print("Affichage des FFT...")
        Title = "Spectre de la tension en sortie des micros (Fondamental à " + str(F) + " Hz)"
        save_plot("Python/Exports/00-FFT_h.png",
                (freqs, fft_h_amp, {'color': Color_h, 'label': "Micro Humbucker - Theorique"}),
                (freqs, fft_h_test_amp, {'marker': '+', 'linestyle': 'None', 'color': Color_h_test, 'label': "Micro Humbucker - Test"}),
                title=Title, xlabel="Frequence (Hz)", ylabel="Tension normalisee")
        print("FFT sauvegardees sous Python/Exports/00-FFT_h.png.")

        save_plot("Python/Exports/00-FFT_s.png",
                (freqs, fft_s_amp, {'color': Color_s, 'label': "Micro Singlecoil - Theorique"}),
                title=Title, xlabel="Frequence (Hz)", ylabel="Tension normalisee")
        print("FFT sauvegardees sous Python/Exports/00-FFT_s.png.")

        save_plot("Python/Exports/00-FFT_th.png",
                (freqs, fft_h_amp, {'color': Color_h, 'label': "Micro Humbucker - Théorique"}),
                (freqs, fft_s_amp, {'color': Color_s, 'label': "Micro Singlecoil - Theorique"}),
                title=Title, xlabel="Frequence (Hz)", ylabel="Tension normalisee")
        print("FFT sauvegardees sous Python/Exports/00-FFT_th.png.")

    print("-" * 100)
    print("\nAffichage des courbes...")

    #* --- Micro Humbucker (corde) ---*#
    print("Micro Humbucker (corde)...")
    save_plot("Python/Exports/01-Humbucker_corde.png",
            (temps, normalize_signal(Test_h), {'marker': '+', 'linestyle': 'None', 'color': Color_h_test, 'label': "Micro Humbucker - Test [V]"}),
            (temps, normalize_signal(fem_h), {'color': Color_h, 'label': "Micro Humbucker - Theorique [V]"}),
            title="Tension induite (normalisee) en sortie du micro Humbucker - Corde seule",
            xlabel="Temps [s]", ylabel="Tension [V]")
    print("Courbes sauvegardees sous Python/Exports/01-Humbucker_corde.png.")

    #* --- Micro Humbucker (Parasite) ---*#
    print("Micro Humbucker (Parasite)...")
    save_plot("Python/Exports/02-Humbucker_parasite.png",
            (temps, Test_h_p, {'marker': '+', 'linestyle': 'None', 'color': Color_h_test, 'label': "Micro Humbucker - Test [V]"}),
            (temps, fem_h_par, {'color': Color_h, 'label': "Micro Humbucker - Theorique [V]"}),
            title="Tension induite en sortie des micros - Parasite seul",
            xlabel="Temps [s]", ylabel="Tension [V]", ylim=(-0.5, 0.5))
    print("Courbes sauvegardees sous Python/Exports/02-Humbucker_parasite.png.")

    #* --- Humbucker (total) ---*#
    print("Micro Humbucker (total)...")
    save_plot("Python/Exports/03-Humbucker_total.png",
            (temps, normalize_signal(Test_h_tot), {'marker': '+', 'linestyle': 'None', 'color': Color_h_test, 'label': "Micro Humbucker - Test [V]"}),
            (temps, normalize_signal(fem_h_tot), {'color': Color_h, 'label': "Micro Humbucker - Theorique [V]"}),
            title="Tension induite (normalisee) en sortie du micro Humbucker - Total",
            xlabel="Temps [s]", ylabel="Tension [V]")
    print("Courbes sauvegardees sous Python/Exports/03-Humbucker_total.png.")

    #* --- Total ---*#
    print("Tous micros total...")
    save_plot("Python/Exports/99-total.png",
            (temps, normalize_signal(Test_h_tot), {'marker': '+', 'linestyle': 'None', 'color': Color_h_test, 'label': "Micro Humbucker - Test [V]"}),
            (temps, normalize_signal(fem_h_tot), {'color': Color_h, 'label': "Micro Humbucker - Theorique [V]"}),
            (temps, normalize_signal(fem_s_tot), {'color': Color_s, 'label': "Micro Singlecoil - Theorique [V]"}),
            title="Tension induite (normalisee) en sortie des micros",
            xlabel="Temps [s]", ylabel="Tension [V]")
    print("Courbes sauvegardees sous Python/Exports/99-total.png.")

    #* --- Comparaison des micros Theorique (corde seule) ---*#
    print("Comparaison des micros  Theorique (corde seule)...")
    plt.figure()
    l3, = plt.plot(temps, normalize_signal(fem_h), color=Color_h, label="Micro Humbucker - Theorique [V]")
    l2, = plt.plot(temps, normalize_signal(fem_s), color=Color_s, label="Micro Singlecoil - Theorique [V]")
    plt.ylabel("Tension induite [V]")
    plt.title("Tension induite (normalisee) en sortie des micros - Corde seule")
    plt.legend(handles=[l3, l2], loc='lower left')
    plt.grid()
    plt.savefig("Python/Exports/Th-Corde.png")
    plt.close()
    print("Courbes sauvegardees sous Python/Exports/Th-Corde.png.")

    #* --- Comparaison des micros Theorique (total) ---*#
    print("Comparaison des micros  Theorique (total)...")
    save_plot("Python/Exports/Th-Total.png",
            (temps, normalize_signal(fem_h_tot), {'color': Color_h, 'label': "Micro Humbucker - Theorique [V]"}),
            (temps, normalize_signal(fem_s_tot), {'color': Color_s, 'label': "Micro Singlecoil - Theorique [V]"}),
            title="Tension induite (normalisee) en sortie des micros",
            xlabel="Temps [s]", ylabel="Tension [V]")
    print("Courbes sauvegardees sous Python/Exports/Th-Total.png.")

    #* --- Droite champ parasite ---*#
    print("Comparaison des micros champ parasite...")
    save_plot("Python/Exports/99-Doite.png",
            (List_Ipar, List_amp_h_th, {'color': Color_h, 'label': "Micro Humbucker - Theorique [V]"}),
            (List_Ipar, List_amp_h_test, {'marker': '+', 'linestyle': 'None', 'color': Color_h_test, 'label': "Micro Humbucker - Test [V]"}),
            (List_Ipar, List_amp_s_th, {'color': Color_s, 'label': "Micro Singlecoil - Theorique [V]"}),
            (List_Ipar, Liste_amp_s_test, {'marker': '+', 'linestyle': 'None', 'color': Color_s_test, 'label': "Micro Singlecoil - Test [V]"}),
            title="Amplitude des tension induite en sortie des micros soumis à un champ parasite",
            xlabel="Amplitude du champ parasite [A]", ylabel="Ampltitude Tension [V]")
    print("Courbes sauvegardees sous Python/Exports/99-Droite.png.")

print("-" * 100)
print("\n---\nFin de l'execution de Modele_fem.py\n---\n")