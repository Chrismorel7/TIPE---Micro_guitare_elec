"""Bibliothèque de modélisation physique pour l'étude des micros de guitare.

Ce module centralise les outils mathématiques et numériques nécessaires à la
caractérisation des micros SingleCoil et Humbucker.
Il permet de simuler la réponse du système depuis la vibration mécanique de la
corde jusqu'au signal électrique filtré.

Fonctionnalités principales :
    - Identification paramétrique (L, C, R_p) par l'algorithme de
        moindre-carre.
    - Calcul des fonctions de transfert H(p) et tracé des diagrammes de
        Bode.
    - Modélisation de la f.e.m. induite e(t) via la loi de Faraday et
        l'approximation dipolaire.
    - Analyse du filtrage en peigne spatial et de la rejection du
        bourdonnement (CMRR).

Auteur: Christophe Morel.
Établissement: Lycée Ferdinand Buisson, CPGE PT.
Projet: TIPE N°39662 - Session 2026.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Physical and geometric constants used by the pickup models.
mu_0 = 4e-7 * np.pi  # Permeabilite magnetique du vide [H.m-1]
mu_r = 3.5  # Permeabilite relative de la corde [.]
K = 0.7

S = 1.5e-4  # Surface d'une bobine [m2]
L_s = 2.4  # Inductance du SingleCoil [H]
N_s = np.sqrt((L_s * 1.5e-2) / (mu_0 * mu_r * S * K))

z_0 = 4e-3  # Hauteur de la corde au repos [m]
L = 0.845  # Longueur de la corde [m]
F = 80  # Frequence fondamentale de la corde [Hz]
mu_c = 9e-3  # Masse linéique de la corde [kg.m-1]
T = 4 * mu_c * (L * F) ** 2  # Tension de la corde [N]
x_s = 0.109  # Position du SingleCoil [m]
Delta = 20e-3  # Ecartement entre les bobinages [m]

chi = 100  # Susceptibilite magnétique de la corde [.]
V = 1e-8  # Volume de la corde magnétisé [m3]
F_B = 50  # Frequence du champ magnétique [Hz]
Bm_max = 0.003  # Champ magnétique du micro à z=0 [T]
Rm = 2.5e-3  # Rayon d'un aimant du micro [m]


def save_plot(
    fig_path: str,
    *plot_calls,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    ylim=None,
    xlim=None,
    legend: bool = True,
    log: bool = False,
    vspan: bool = False,
    error_bars: bool = False,
):
    """
    Enregistre une figure matplotlib avec des options de personnalisation.
    Args:
        fig_path (str): Chemin vers le fichier de sauvegarde
        plot_calls (tuple): Appels de tracé sous la forme (x, y, kwargs)
        title (str): Titre de la figure
        xlabel (str): nom de l'axe des x
        ylabel (str): nom de l'axe des y
        xlim (tuple): Limites de l'axe des x
        ylim (tuple): Limites de l'axe des y
        legend (bool): presence de la legende
        log (bool): presence d'une echelle logarithmique sur l'axe des x
        vspan (bool): presence des zones de frequences fondamentales et harmoniques
        error_bars (bool): active l'utilisation de plt.errorbar (necessite yerr/xerr dans kwargs)
    """
    output_path = Path(fig_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    handles_curves = []
    z_handles = []

    if vspan:
        z2 = plt.axvspan(
            82.41,
            1174.66,
            color="#ADD8E6",
            alpha=0.6,
            label="Frequences fondamentales",
            zorder=0,
        )
        z1 = plt.axvspan(
            1174.66,
            5 * 1174.66,
            facecolor="none",
            alpha=0.5,
            hatch="////",
            edgecolor="#424242",
            label="Frequences harmoniques",
            zorder=0,
        )
        z_handles = [z1, z2]

    for call in plot_calls:
        x, y, kwargs = call
        if error_bars:
            # plt.errorbar retourne un ErrorbarContainer compatible avec legend()
            line = plt.errorbar(x, y, **kwargs)
            handles_curves.append(line)
        else:
            (line,) = plt.plot(x, y, **kwargs)
            handles_curves.append(line)

    if title:
        plt.title(title, fontsize=12, fontweight="bold")
    if log:
        plt.xscale("log")
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)

    plt.grid(True, alpha=0.6)

    if legend:
        leg_courbes = plt.legend(handles=handles_curves, loc="best", framealpha=1)
        if vspan:
            plt.gca().add_artist(leg_courbes)
            plt.legend(
                handles=z_handles,
                title="Zones d'etude",
                bbox_to_anchor=(0.5, -0.18),
                loc="upper center",
                ncol=2,
                borderaxespad=0.0,
            )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def normalize_signal(signal: np.ndarray):
    """
    Normalise un signal entre -1 et 1.

    :param signal: Signal à normaliser
    :type signal: np.ndarray
    """
    if signal.size == 0:
        raise ValueError("signal must contain at least one sample")
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val


def compute_and_normalize_fft(signal: np.ndarray, sample_count: int):
    """
    Calcule et normalise l'amplitude du spectre de Fourier d'un signal.
    Args:
        signal (np.ndarray): Signal a analyser
        sample_count (int): Number of samples used in the FFT.
    Returns:
        amp: Signal numpy fft
    """
    if signal.size == 0 or sample_count <= 0:
        raise ValueError("signal and sample_count must be positive")
    if sample_count != signal.size:
        raise ValueError("sample_count must match the signal length")
    spectrum = np.fft.fft(signal)
    return np.abs(spectrum) / sample_count


def extraire_pics(frequences, amplitudes, prominence=0.05, distance=10):
    """
    Extrait les pics significatifs d'un spectre de Fourier en utilisant la proeminence.
    Args:
        frequences (np.ndarray): Tableau des frequences
        amplitudes (np.ndarray): Tableau des amplitudes correspondantes
        prominence (float): Hauteur relative du pic par rapport à sa base locale.
                            Permet de detecter les petits pics aigus sans prendre le bruit fort.
        distance (int): Distance minimale entre deux pics pour les considérer comme séparés.
    Returns:
        L_freq: Liste des frequences
        L_amp: Liste des amplitudes
    """
    if frequences.size != amplitudes.size:
        raise ValueError("frequences and amplitudes must have the same length")
    if distance < 1 or prominence < 0:
        raise ValueError("distance must be positive and prominence non-negative")

    indices, _ = find_peaks(amplitudes, prominence=prominence, distance=distance)

    # Extraction et conversion en listes
    liste_frequences = frequences[indices].tolist()
    liste_amplitudes = amplitudes[indices].tolist()

    return np.array(liste_frequences), np.round(liste_amplitudes, 3)


def pickup_impedance(
    parameters: np.ndarray, series_resistance: float, angular_frequency: np.ndarray
) -> np.ndarray:
    """Calculate the impedance magnitude for a pickup equivalent circuit.

    Args:
        parameters: Inductance [H], parallel capacitance [nF], and parallel
            resistance [kOhm].
        series_resistance: Measured DC resistance [ohm].
        angular_frequency: Angular frequencies [rad/s].

    Returns:
        The impedance magnitude at each angular frequency [ohm].
    """
    if parameters.shape != (3,) or np.any(parameters <= 0):
        raise ValueError("parameters must contain three positive circuit values")
    if series_resistance <= 0:
        raise ValueError("series_resistance must be positive")
    inductance, capacitance_nf, parallel_resistance_kohm = parameters
    capacitance = capacitance_nf * 1e-9
    parallel_resistance = parallel_resistance_kohm * 1e3
    denominator = series_resistance**2 + (inductance * angular_frequency) ** 2
    admittance_real = (1 / parallel_resistance) + (series_resistance / denominator)
    admittance_imag = (capacitance * angular_frequency) - (
        inductance * angular_frequency / denominator
    )
    return 1 / np.sqrt(admittance_real**2 + admittance_imag**2)


def matrice_jacobienne(X: np.ndarray, rs: float, f: np.ndarray) -> np.ndarray:
    """Basee sur l'algorithme de moindres_carres, cette fonction retourne la matrice Jacobienne J
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
    den_L = rs**2 + (L * w) ** 2
    Y_re = (1 / Rp) + (rs / den_L)
    Y_im = (C * w) - (L * w / den_L)
    Z = 1 / np.sqrt(Y_re**2 + Y_im**2)
    Z3 = Z**3

    # Derivees partielles de Y
    dYre_dL = -(rs * 2 * L * w**2) / (den_L**2)
    dYim_dL = -(w * (rs**2 - (L * w) ** 2)) / (den_L**2)
    dZ_dL = -Z3 * (Y_re * dYre_dL + Y_im * dYim_dL)

    dYim_dCn = w * 1e-9
    dZ_dCn = -Z3 * (Y_im * dYim_dCn)

    dYre_dRpn = -1 / (1000 * Rp_k**2)
    dZ_dRpk = -Z3 * (Y_re * dYre_dRpn)

    return np.column_stack((dZ_dL, dZ_dCn, dZ_dRpk))


def residus(X: np.ndarray, rs: float, f: np.ndarray, Z_exp: np.ndarray) -> np.ndarray:
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
    return Z_exp - pickup_impedance(X, rs, 2 * np.pi * f)


def moindres_carres(
    X_init: np.ndarray,
    rs: float,
    f: np.ndarray,
    Z_exp: np.ndarray,
    N: int,
    learning_rate: float,
):
    """algorithme de moindre-carre pour l'optimisation non-lineaire des parametres electriques du micro

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
    if N < 0 or learning_rate <= 0:
        raise ValueError("N must be non-negative and learning_rate must be positive")
    X = np.array(X_init, dtype=float)
    if X.shape != (3,) or np.any(X <= 0):
        raise ValueError("X_init must contain three positive circuit parameters")
    if f.size != Z_exp.size or f.size == 0:
        raise ValueError("f and Z_exp must be non-empty arrays of equal length")
    alpha = learning_rate
    v = 3

    for _ in range(N):
        r = residus(X, rs, f, Z_exp)
        eps = np.sum(r**2)

        # 1. Calcul de la Jacobienne classique
        J = matrice_jacobienne(X, rs, f)

        # 2. NORMALISATION (Adimensionnement)
        # On multiplie chaque colonne de J par la valeur actuelle du paramètre X correspondant.
        # Cela permet de calculer des dérivées relatives (sensibilité à une variation en %)
        J_norm = J * X

        # 3. Calcul de la matrice Hessienne et du gradient normalisés
        H = J_norm.T @ J_norm
        g = J_norm.T @ r

        # Amortissement de Marquardt
        H_lm = H + alpha * np.diag(np.diag(H) + 1e-8)

        try:
            # delta_rel contient maintenant les variations relatives (ex: 0.05 = +5%)
            delta_rel = np.linalg.solve(H_lm, g)
        except np.linalg.LinAlgError:
            break

        # 4. Mise à jour des paramètres (X_new = X + X * delta_rel)
        X_new = X * (1 + delta_rel)

        # Contrainte physique : paramètres toujours positifs
        X_new = np.maximum(X_new, [1e-2, 1e-2, 1e-2])

        eps_new = np.sum(residus(X_new, rs, f, Z_exp) ** 2)

        if eps_new < eps:
            alpha /= v
            X = X_new
            # Condition d'arrêt sur la norme des variations relatives
            if np.linalg.norm(delta_rel) < 1e-7:
                break
        else:
            alpha *= v

    return X, np.sum(residus(X, rs, f, Z_exp) ** 2)


def calcul_bode(
    X: np.ndarray, rs: float, f: np.ndarray, Ramp: float = 1e6
) -> tuple[np.ndarray, np.ndarray]:
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

    # Impedance serie
    Z_ser = rs + 1j * L * w

    # Impedance parallèle
    Y_para = (1 / Rp) + (1 / Ramp) + (1j * C * w)
    Z_para = 1 / Y_para

    # Fonction de transfert
    H = Z_para / (Z_ser + Z_para)

    gain = 20 * np.log10(np.abs(H))
    phase = np.angle(H, deg=True)
    return gain, phase


def entree(n: int, x: float, t: np.ndarray, A0: float, x0: float) -> np.ndarray:
    """Calculate the string displacement at a position over time.

    Args:
        n (int): Nombre de modes consideres
        x (float): Position a observer
        t (np.ndarray): Tableau des instants
        A0 (float): Amplitude initiale de la vibration
        x0 (float): Position de la perturbation

    Returns:
        np.ndarray: Signal de vibration de la corde
    """
    if n < 1 or not 0 < x0 < L or not 0 <= x <= L:
        raise ValueError("n must be positive and x and x0 must lie on the string")
    s = np.zeros_like(t, dtype=float)
    v = np.sqrt(T / mu_c)
    for i in range(1, n + 1):
        Bn = (
            (A0 * L**2.5 * np.sqrt(2))
            / (i**2 * np.pi**2 * x0 * (L - x0))
            * np.sin(i * np.pi * x0 / L)
        )
        s += Bn * np.sin(i * np.pi * x / L) * np.cos(i * np.pi * v * t / L)
    return s


def fem_singlecoil(n: int, t: np.ndarray, A0: float, x0: float) -> np.ndarray:
    """Calculate the induced voltage in a single-coil pickup.

    Args:
        n: Number of string vibration modes.
        t: Sample times [s].
        A0: Initial string displacement [m].
        x0: Pluck position [m].

    Returns:
        Induced voltage samples [V].
    """

    # 1. Calcul des trajectoires vibratoires (Position et Vitesse)
    zs_t = np.full_like(t, z_0)
    vs_t = np.zeros_like(t)
    omega1 = 2 * np.pi * F

    for i in range(1, n + 1):
        Bn = (
            (A0 * L**2.5 * np.sqrt(2))
            / (i**2 * np.pi**2 * x0 * (L - x0))
            * np.sin(i * np.pi * x0 / L)
        )

        omega_i = i * omega1
        spatial_node = np.sin(i * np.pi * x_s / L)

        zs_t += Bn * spatial_node * np.cos(omega_i * t)
        vs_t -= Bn * omega_i * spatial_node * np.sin(omega_i * t)

    # 2. Modélisation du champ magnétique et de son gradient
    B_ext = Bm_max * (Rm**3 / (Rm**2 + zs_t**2) ** 1.5)
    dB_dz = -3 * Bm_max * Rm**3 * zs_t / (Rm**2 + zs_t**2) ** 2.5

    # 3. Calcul du terme de transduction non-linéaire
    transduction_gain = (3 * B_ext / zs_t**4) - (1 / zs_t**3 * dB_dz)

    # 4. Calcul de la f.e.m. finale
    constante_flux = (N_s * S * chi * V) / (2 * np.pi)

    return constante_flux * transduction_gain * vs_t


def fem_humbucker(
    n: int, N: int, x1: float, x2: float, t: np.ndarray, A0: float, x0: float
) -> np.ndarray:
    """Force electromotrice induite dans le micro Humbucker (Théorie Robuste Non-Linéaire)
    Args:
        n (int): Nombre de modes consideres
        N (int): Nombre de spires par bobine
        x1 (float): Position du 1er bobinage sur la corde
        x2 (float): Position du 2eme bobinage sur la corde
        t (np.ndarray): Tableau des instants
        A0 (float): Amplitude initiale de la vibration
        x0 (float): Position de la perturbation
    Returns:
        np.ndarray: Signal de la f.e.m. induite totale (V)
    """

    def calculate_coil_fem(x_pos):
        # 1. Calcul des trajectoires vibratoires (Position et Vitesse) à la position x_pos
        zs_t = np.full_like(t, z_0)
        vs_t = np.zeros_like(t)
        omega1 = 2 * np.pi * F

        for i in range(1, n + 1):
            Bn = (
                (A0 * L**2.5 * np.sqrt(2))
                / (i**2 * np.pi**2 * x0 * (L - x0))
                * np.sin(i * np.pi * x0 / L)
            )

            omega_i = i * omega1
            spatial_node = np.sin(i * np.pi * x_pos / L)

            zs_t += Bn * spatial_node * np.cos(omega_i * t)
            vs_t -= Bn * omega_i * spatial_node * np.sin(omega_i * t)

        # 2. Modélisation du champ magnétique et de son gradient
        B_ext = Bm_max * (Rm**3 / (Rm**2 + zs_t**2) ** 1.5)
        dB_dz = -3 * Bm_max * Rm**3 * zs_t / (Rm**2 + zs_t**2) ** 2.5

        # 3. Calcul du terme de transduction non-linéaire G(z)
        transduction_gain = (3 * B_ext / zs_t**4) - (1 / zs_t**3 * dB_dz)

        # 4. Calcul de la f.e.m. élémentaire
        constante_flux = (N * S * chi * V) / (2 * np.pi)
        return constante_flux * transduction_gain * vs_t

    # Le Humbucker additionne les signaux utiles des deux bobines car
    # l'inversion de polarité magnétique compense l'inversion du bobinage
    return calculate_coil_fem(x1) + calculate_coil_fem(x2)


def attenuation_theorique_db(
    frequences: np.ndarray, x1: float, x2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcule la Fonction de Transfert Relative (Attenuation) du Humbucker
    par rapport au Single Coil en fonction de la frequence.
    Modèle : Filtrage en peigne dû à l'ecartement des bobines.
    Args:
        frequences (np.ndarray): Tableau des fréquences
        xm (float): Position du micro SingleCoil
        x1 (float): Position du 1er bobinage du micro humbucker sur la corde
        x2 (float): Position du 2ème bobinage du micro humbucker sur la corde
    Returns:
        tuple[np.ndarray, np.ndarray]: fonction de transfert en dB du Humbucker par rapport au Single Coil, et atténuation due au filtre peigne seul
    """
    v = 2 * L * F
    k = 2 * np.pi * frequences / v
    Delta = abs(x2 - x1)
    x_humbucker = (x2 + x1) / 2

    signal_sc = np.sin(k * x_s)
    Peigne_hb = np.cos(k * Delta / 2)
    signal_hb = 2 * Peigne_hb * np.sin(k * x_humbucker)

    return signal_hb, Peigne_hb, signal_sc


def helmholtz_field(
    position: float, current: float, turns: float, radius: float
) -> float:
    """Amplitude du champ magnétique dans un système de Bobine de Helmotz

    Args:
        position (float): Position along the coil axis.
        current (float): Current through the coils.
        turns (float): Number of turns per coil.
        radius (float): Coil radius.

    Returns:
        float: Amplitude du champ magnétique
    """
    field = (
        (turns * mu_0 * radius**2)
        / 2
        * (
            1 / (radius**2 + (position - radius / 2) ** 2) ** (3 / 2)
            + 1 / (radius**2 + (position + radius / 2) ** 2) ** (3 / 2)
        )
        * current
    )
    return field


def fem_par_s(t: np.ndarray, B0: float, phi: float) -> np.ndarray:
    """Force electromotrice induite dans une bobine par une variation sinusoïdale d'un champ magnétique exterieur
    Args:
        t (np.ndarray): Tableau des instants
        B0 (float): Amplitude du champ magnétique
        phi (float): Phase initiale de la variation du champ magnétique
    Returns:
        np.ndarray: Tableau du signal de la force electromotrice induite dans une bobine
    """
    return N_s * S * 2 * np.pi * F_B * B0 * np.sin(2 * np.pi * F_B * t + phi) * mu_r


def fem_par_h(
    N: int, x1: float, x2: float, t: np.ndarray, B0: float, alpha: float
) -> np.ndarray:
    """Force electromotrice induite dans un humbucker par une variation sinusoïdale d'un champ magnétique exterieur
    Args:
        N (int): Nombre de spires du micro
        x1 (float): Position du 1er bobinage du micro humbucker sur la corde
        x2 (float): Position du 2ème bobinage du micro humbucker sur la corde
        t (np.ndarray): Tableau des instants
        B0 (float): Amplitude du champ magnétique
        alpha (float): Gradient du champ magnétique
    Returns:
        np.ndarray: Tableau du signal de la force electromotrice induite dans un humbucker
    """
    return (
        N
        * S
        * 2
        * np.pi
        * F_B
        * B0
        * alpha
        * abs(x1 - x2)
        * np.sin(2 * np.pi * F_B * t)
        * mu_r
    )
