"""Fichier contenant toute la modelisation du TIPE"""

"""Importation des bibilothèques"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

"""Constantes globales"""
rho = 1.68 * 10**(-8)   # Masse volumique du cuivre (Ohm.m)
mu0 = 4e-7 * np.pi      # Permeabilite magnetique du vide (H.m-1)
eps0 = 8.85e-12         # Permittivite du vide (F.m-1)
eps = 1e-2


"""Partie 1 : Fonctions Utilitaires
    """
def save_plot(fig_path: str, *plot_calls, title:str="", xlabel:str="", ylabel:str="", ylim=None, xlim=None, legend:bool=True, log:bool=False, vspan:bool=False):
    """
    Enregistre une figure matplotlib avec des options de personnalisation.

    :param fig_path: Chemin vers le fichier de sauvegarde
    :type fig_path: str
    :param plot_calls: Appels de tracé sous la forme (x, y, kwargs)
    :type plot_calls: tuple
    :param title: Titre de la figure
    :type title: str
    :param xlabel: nom de l'axe des x
    :type xlabel: str
    :param ylabel: nom de l'axe des y
    :type ylabel: str
    :param ylim: Limites de l'axe des y
    :type ylim: tuple
    :param legend: presence de la legende
    :type legend: bool
    :param log: presence d'une echelle logarithmique sur l'axe des x
    :type log: bool
    :param vspan: presence des zones de frequences fondamentales et harmoniques
    :type vspan: bool
    """
    plt.figure(figsize=(6, 5)) 
    handles_curves = []
    z_handles = []
    if vspan:
        z2 = plt.axvspan(82.41, 1174.66, color="#F0E68C", alpha=0.3, 
                        label="Frequences fondamentales", zorder=0)
        z1 = plt.axvspan(1174.66, 5*1174.66, facecolor="none", alpha=0.5, hatch="////", 
                        edgecolor="#424242", label="Frequences harmoniques", zorder=0)
        z_handles = [z1, z2]
    
    for call in plot_calls:
        x, y, kwargs = call
        # On recupère l'objet ligne retourne par plot (c'est un tuple, d'où le [0])
        line, = plt.plot(x, y, **kwargs)
        handles_curves.append(line)
    
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if ylim is not None: plt.ylim(ylim)
    if xlim is not None: plt.xlim(xlim)
    
    plt.grid(True, alpha=0.6) # Grid plus discrète pour le contraste
    if log: plt.xscale('log')

    if legend:
        # 1. Legende des courbes (Interne)
        leg_courbes = plt.legend(handles=handles_curves, loc='best', framealpha=1)
        
        # 2. Legende des zones (Externe à droite)
        if vspan:
            plt.gca().add_artist(leg_courbes)
            plt.legend(handles=z_handles, title="Zones d'etude", bbox_to_anchor=(0.5, -0.18), loc='upper center', ncol=2, borderaxespad=0.)
    
    # Ajustement automatique pour ne pas couper la legende externe
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300) # dpi=200 pour une nettete accrue en projection
    plt.close()

def normalize_signal(signal: np.ndarray):
    """
    Normalise un signal entre -1 et 1.
    
    :param signal: Signal à normaliser
    :type signal: np.ndarray
    """
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val

def compute_and_normalize_fft(signal: np.ndarray, Ne_fft: int):
    """
    Calcule et normalise l'amplitude du spectre de Fourier d'un signal.

    :param signal: Signal à analyser
    :type signal: np.ndarray
    :param Ne_fft: Nombre de points dans la FFT
    :type Ne_fft: int
    """
    
    fft = np.fft.fft(signal)
    amp = np.abs(fft) / Ne_fft
    return amp

def extraire_pics_separes(frequences, amplitudes, prominence=0.05, distance=10):
    """
    Extrait les pics significatifs d'un spectre de Fourier en utilisant la proeminence.
    Args:
        frequences (np.ndarray): Tableau des frequences
        amplitudes (np.ndarray): Tableau des amplitudes correspondantes
        prominence (float): Hauteur relative du pic par rapport à sa base locale.
                            Permet de detecter les petits pics aigus sans prendre le bruit fort.
        distance (int): Distance minimale entre deux pics pour les considérer comme séparés.
        
    """
    # Utilisation de 'prominence' au lieu de 'height' pour le problème d'amplitude variable
    indices, _ = find_peaks(amplitudes, prominence=prominence, distance=distance)
    
    # Extraction et conversion en listes
    liste_frequences = frequences[indices].tolist()
    liste_amplitudes = amplitudes[indices].tolist()
    
    return np.array(liste_frequences), np.round(liste_amplitudes, 3)

"""Partie 2 : Modelisation Electrique
    """

def Z_micro(X:np.ndarray, rs:float, w:np.ndarray)-> np.ndarray:
    """Calcul l'impedance d'un micro de guitare en une pulsation w

    Args:
        X (np.ndarray): Vecteur des caracteristique electrique interne du micro [L, C, Rp] avec L l'inductance [H], C la capacite parallèle [nF]et
            Rp la resistance parallèle [kOhm] modelisant les courants de foucauts
        rs (float): Resistance serie (DC) du micro. Peut etre mesure avec un Ohmmetre 
        w (np.ndarray): Tableau des pulsation w

    Returns:
        np.darray: Tableau Numpy contenant le module de l'impedance complexe |Z| en chaque pulsation w 
    """
    L, C_n, Rp_k = X
    C, Rp = C_n * 1e-9, Rp_k * 1e3
    
    # Parties reelle et imaginaire de l'admittance Y
    den_L = rs**2 + (L * w)**2
    Y_re = (1 / Rp) + (rs / den_L)
    Y_im = (C * w) - (L * w / den_L)
    
    return 1 / np.sqrt(Y_re**2 + Y_im**2)

def Matrice_Jacobienne(X:np.ndarray, rs:float, f:np.ndarray)->np.ndarray:
    """Basee sur l'algorythme de Levenberg_Marquardt, cette fonction retourne la matrice Jacobienne J

    Args:
        X (np.ndarray): Vecteur des caracteristique electrique interne du micro [L, C, Rp] avec L l'inductance [H], C la capacite parallèle [nF]et
            Rp la resistance parallèle [kOhm] modelisant les courants de foucauts
        rs (float):Resistance serie (DC) du micro. Peut etre mesure avec un Ohmmetre
        f (np.ndarray): Tableau des frequence f

    Returns:
        np.ndarray: Matrice Jacobienne definie tel que chaque colonne i contient les derivees partielles par rapport a la variable situe en i dans X pour toute
            pulsation w
    """
    w = 2 * np.pi * f
    L, C_n, Rp_k = X
    C, Rp = C_n * 1e-9, Rp_k * 1e3
    
    # Variables intermediaires
    den_L = rs**2 + (L * w)**2
    Y_re = (1 / Rp) + (rs / den_L)
    Y_im = (C * w) - (L * w / den_L)
    Z = 1 / np.sqrt(Y_re**2 + Y_im**2)
    Z3 = Z**3

    # Derivees partielles de Y
    dYre_dL = - (rs * 2 * L * w**2) / (den_L**2)
    dYim_dL = - (w * (rs**2 - (L * w)**2)) / (den_L**2)
    dZ_dL = - Z3 * (Y_re * dYre_dL + Y_im * dYim_dL)
    
    dYim_dCn = w * 1e-9
    dZ_dCn = - Z3 * (Y_im * dYim_dCn)

    dYre_dRpn = - 1 / (1000 * Rp_k**2)
    dZ_dRpk = - Z3 * (Y_re * dYre_dRpn)

    return np.column_stack((dZ_dL, dZ_dCn, dZ_dRpk))

def residus(X:np.ndarray, rs:float, f:np.ndarray, Z_exp:np.ndarray)-> np.ndarray:
    """Calcul l'ecart (l'erreur) entre l'impedance calcule theoriquement et l'impedance reelle experimentale

    Args:
        X (np.ndarray): Vecteur des caracteristique electrique interne du micro [L, C, Rp] avec L l'inductance [H], C la capacite parallèle [nF]et
            Rp la resistance parallèle [kOhm] modelisant les courants de foucauts
        rs (float):Resistance serie (DC) du micro. Peut etre mesure avec un Ohmmetre
        f (np.ndarray): Tableau des frequence f
        Z_exp (np.ndarray): Tableau Numpy des impedances experimentale releve en Ohm

    Returns:
        np.ndarray: Tableau Numpy des distances separant chaque point theorique au point experimentaux
    """
    return Z_exp - Z_micro(X, rs, 2 * np.pi * f)

def Levenberg_Marquardt(X_init:np.ndarray, rs:float, f:np.ndarray, Z_exp:np.ndarray, N:int, learning_rate:float):
    """Algorythme de Levenberg-Marquardt pour l'optimisation non-lineaire des parametres electriques du micro

    Args:
        X_init (np.ndarray): Vecteur initial des caracteristique electrique interne du micro [L, C, Rp] avec L l'inductance [H], C la capacite parallèle [nF]et
            Rp la resistance parallèle [kOhm] modelisant les courants de foucauts
        rs (float): Resistance serie (DC) du micro. Peut etre mesure avec un Ohmmetre
        f (np.ndarray): Tableau des frequence f
        Z_exp (np.ndarray): Tableau Numpy des impedances experimentale releve en Ohm
        N (int): Nombre d'iterations de l'algorithme

    Returns:
        np.ndarray: Vecteur optimisé des caracteristique electrique interne du micro [L, C, Rp] avec L l'inductance [H], C la capacite parallèle [nF]et
            Rp la resistance parallèle [kOhm] modelisant les courants de foucauts
    """
    X = np.array(X_init, dtype=float)
    alpha = learning_rate
    v = 3
    r = None
    eps = None
    
    for _ in range(N):
        r = residus(X, rs, f, Z_exp)
        eps = np.sum(r**2)
        J = Matrice_Jacobienne(X, rs, f)

        H = J.T @ J
        g = J.T @ r
        
        # Amortissement de Marquardt
        H_lm = H + alpha * np.diag(np.diag(H) + 1e-8)
        
        try:
            delta = np.linalg.solve(H_lm, g)
        except np.linalg.LinAlgError:
            break

        X_new = X + delta
        # Contrainte physique : paramètres toujours positifs
        X_new = np.maximum(X_new, [1e-3, 1e-3, 1e-3]) 
        eps_new = np.sum(residus(X_new, rs, f, Z_exp)**2)
        if eps_new < eps:
            alpha /= v
            X = X_new
            if np.linalg.norm(delta) / np.linalg.norm(X) < 1e-12:
                break
        else:
            alpha *= v
            
    return X, eps_new

def calcul_bode(X:np.ndarray, rs:float, f:np.ndarray, Ramp:float=1e6)-> tuple[np.ndarray, np.ndarray]:
    """Calcul la fonction de transfert en gain et phase d'un micro de guitare en fonction de ses parametres internes

    Args:
        X (np.ndarray): Vecteur optimisé des caracteristique electrique interne du micro [L, C, Rp] avec L l'inductance [H], C la capacite parallèle [nF]et
            Rp la resistance parallèle [kOhm] modelisant les courants de foucauts
        rs (float): Resistance serie (DC) du micro. Peut etre mesure avec un Ohmmetre
        f (np.ndarray): Tableau des frequence f
        Ramp (float, optional): Resistance d'entree de l'amplificateur. Defaults to 1e6.

    Returns:
        tuple[np.ndarray, np.ndarray]: Deux tableaux Numpy contenant respectivement le gain (en dB) et la phase (en degres) de la fonction de transfert du micro
    """
    w = 2 * np.pi * f
    L, C, Rp = X[0], X[1] * 1e-9, X[2] * 1e3
    
    # Impedance serie (la bobine qui genère le signal)
    Z_ser = rs + 1j * L * w
    
    # Impedance parallèle (ce qui "shunte" le signal vers la masse)
    Y_para = (1 / Rp) + (1 / Ramp) + (1j * C * w)
    Z_para = 1 / Y_para
    
    # Fonction de transfert correcte
    H = Z_para / (Z_ser + Z_para)
    
    gain = 20 * np.log10(np.abs(H))
    phase = np.angle(H, deg=True)
    return gain, phase


"""Partie 3 : Modelisation Acoustique
    """

def entree(n: int, x: float, t: np.ndarray, A0: float, L: float, T: float, mu: float, x0: float) -> np.ndarray:
    """Signal de vibration de la corde

    Args:
        n (int): Nombre de modes consideres
        x (float): Position a observer
        t (np.ndarray): Tableau des instants
        A0 (float): Amplitude initiale de la vibration
        L (float): Longueur de la corde
        T (float): Tension de la corde
        mu (float): Masse linéique de la corde
        x0 (float): Position de la perturbation

    Returns:
        np.ndarray: Signal de vibration de la corde
    """
    s = 0
    v = np.sqrt(T / mu)
    for i in range(1, n + 1):
        Bn = (A0 * L**2.5 * np.sqrt(2)) / (i**2 * np.pi**2 * x0 * (L - x0)) * np.sin(i * np.pi * x0 / L)
        s += Bn * np.sin(i * np.pi * x / L) * np.cos(i * np.pi * v * t / L)
    return s

def fem_singlecoil(n: int, N: int, xm: float, t: np.ndarray, A0: float, L: float, T: float, mu: float, x0: float, S: float, dBdy: float) -> np.ndarray:
    """Force electromotrice induite dans le micro singlecoil

    Args:
        n (int): Nombre de modes consideres
        N (int): Nombre de spires de la bobine du micro singlecoil
        xm (float): Position du micro sur la corde
        t (np.ndarray): Tableau des instants
        A0 (float): Amplitude initiale de la vibration
        L (float): Longueur de la corde
        T (float): Tension de la corde
        mu (float): Masse linéique de la corde
        x0 (float): Position de la perturbation
        S (float): Surface effective du micro
        dBdy (float): Variation du flux magnétique par unité de déplacement vertical

    Returns:
        np.ndarray: Tableau du signal de la force electromotrice induite dans le micro singlecoil
    """
    s = 0
    for i in range(1, n + 1):
        fn = (i / (2 * L)) * np.sqrt(T / mu)
        An = 2 * np.pi * fn * np.sin(i * np.pi * xm / L)
        Bn = (A0 * L**2.5 * np.sqrt(2)) / (i**2 * np.pi**2 * x0 * (L - x0)) * np.sin(i * np.pi * x0 / L)
        s += An * Bn * np.sin(2 * np.pi * fn * t)
    return N * S * dBdy * s

def fem_humbucker(n: int, N: int, x1: float, x2: float, t: np.ndarray, A0: float, L: float, T: float, mu: float, x0: float, S: float, dBdy: float) -> np.ndarray:
    """Force electromotrice induite dans le micro Humbucker

    Args:
        n (int): Nombre de modes consideres
        N (int): Nombre de spires de la bobine du micro Humbucker
        x1 (float): Position du 1er bobinage du micro sur la corde
        x2 (float): Position du 2ème bobinage du micro sur la corde
        t (np.ndarray): Tableau des instants
        A0 (float): Amplitude initiale de la vibration
        L (float): Longueur de la corde
        T (float): Tension de la corde
        mu (float): Masse linéique de la corde
        x0 (float): Position de la perturbation
        S (float): Surface effective du micro
        dBdy (float): Variation du flux magnétique par unité de déplacement vertical

    Returns:
        np.ndarray: Tableau du signal de la force electromotrice induite dans le micro Humbucker
    """
    s = 0
    for i in range(1, n + 1):
        fn = (i / (2 * L)) * np.sqrt(T / mu)
        An = 2 * np.pi * fn * (np.sin(i * np.pi * x1 / L) + np.sin(i * np.pi * x2 / L))
        Bn = (A0 * L**2.5 * np.sqrt(2)) / (i**2 * np.pi**2 * x0 * (L - x0)) * np.sin(i * np.pi * x0 / L)
        s += An * Bn * np.sin(2 * np.pi * fn * t)
    return N * S * dBdy * s

def attenuation_theorique_db(frequences: np.ndarray, x1: float, x2: float, xm: float, L: float, F: float) -> tuple[np.ndarray, np.ndarray]:
    """Calcule la Fonction de Transfert Relative (Attenuation) du Humbucker 
    par rapport au Single Coil en fonction de la frequence.
    
    Modèle : Filtrage en peigne dû à l'ecartement des bobines.

    Args:
        frequences (np.ndarray): Tableau des fréquences
        x1 (float): Position du 1er bobinage du micro humbucker sur la corde
        x2 (float): Position du 2ème bobinage du micro humbucker sur la corde
        xm (float): Position du micro singlecoil sur la corde
        L (float): Longueur de la corde
        F (float): Fréquence fondamentale de la corde

    Returns:
        tuple[np.ndarray, np.ndarray]: fonction de transfert en dB du Humbucker par rapport au Single Coil, et atténuation due au filtre peigne seul
    """
    v = 2 * L * F
    k = 2 * np.pi * frequences / v
    delta_x = abs(x2-x1)
    x_humbucker = (x2+x1)/2
    
    signal_sc = np.abs(np.sin(k * xm))

    # Signal HB (Position + Filtre Peigne)
    # Note : x_humbucker est le centre du micro
    filtre_peigne = np.cos(k * delta_x / 2)
    signal_hb = np.abs(np.sin(k * x_humbucker)) * np.abs(filtre_peigne)

    # Ratio Theorique (Avec gestion de la division par zero)
    ratio = np.divide(signal_hb, signal_sc, out=np.zeros_like(signal_sc), where=signal_sc!=0)
    ratio_db = 20 * np.log10(ratio + 1e-6)
    return ratio_db, 20*np.log10(np.abs(np.cos(k*delta_x/2)))

def B(x: float, I0: float, Nbob: float, Rbob: float) -> float:
    """Amplitude du champ magnétique dans un système de Bobine de Helmotz

    Args:
        x (float): position le long de l'axe des bobines
        I0 (float): Intensité du courant dans les bobines
        Nbob (float): Nombre de spires par bobine
        Rbob (float): Rayon des bobines

    Returns:
        float: Amplitude du champ magnétique
    """
    B0 = (Nbob * mu0 * Rbob**2) / 2 * (1 / (Rbob**2 + (x - Rbob / 2)**2)**(3 / 2) + 1 / (Rbob**2 + (x + Rbob / 2)**2)**(3 / 2)) * I0 
    return B0

def fem_par_s(N: int, fB: float, t: np.ndarray, B0: float, S: float, phi: float) -> np.ndarray:
    """Force electromotrice induite dans une bobine par une variation sinusoïdale d'un champ magnétique exterieur

    Args:
        N (int): Nombre de spires du micro
        fB (float): Fréquence de variation du champ magnétique
        t (np.ndarray): Tableau des instants
        B0 (float): Amplitude du champ magnétique
        S (float): Surface effective du micro
        phi (float): Phase initiale de la variation du champ magnétique

    Returns:
        np.ndarray: Tableau du signal de la force electromotrice induite dans une bobine

    Returns:
        np.ndarray: _description_
    """
    return N*S*2*np.pi*fB*B0*np.sin(2*np.pi*fB*t + phi)

def fem_par_h(N: int, x1: float, x2: float, fB: float, t: np.ndarray, B0: float, alpha: float, S: float) -> np.ndarray:
    """Force electromotrice induite dans un humbucker par une variation sinusoïdale d'un champ magnétique exterieur

    Args:
        N (int): Nombre de spires du micro
        x1 (float): Position du 1er bobinage du micro humbucker sur la corde
        x2 (float): Position du 2ème bobinage du micro humbucker sur la corde
        fB (float): Fréquence de variation du champ magnétique
        t (np.ndarray): Tableau des instants
        B0 (float): Amplitude du champ magnétique
        alpha (float): Gradient du champ magnétique
        S (float): Surface effective du micro

    Returns:
        np.ndarray: Tableau du signal de la force electromotrice induite dans un humbucker
    """
    return N*S*2*np.pi*fB*B0*alpha*abs(x1-x2)*np.sin(2*np.pi*fB*t)
