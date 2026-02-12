import numpy as np
import matplotlib.pyplot as plt

def model_guitar_pickup():
    # --- 1. PARAMÈTRES (À AJUSTER) ---
    fs = 44100              # Fréquence d'échantillonnage (Hz)
    duration = 0.1          # Durée de la simulation (s)
    f0 = 78              # Fréquence de la corde (Sol / G3) (Hz)
    z0 = 3.5e-3             # Distance corde-micro au repos (m)
    xm = 0.11
    A0 = 3e-3
    L = 0.845
    mu = 9 * 10**(-3)       # Masse lineique de la corde
    T = 4 * mu * (L * f0)**2    #? Tension de la corde
    x0 = 0.140                 #? Origine de la perturbation

    
    # Dimensions du plot (aimant rectangulaire 2a x 2b)
    a = 2.4e-3              # Demi-largeur (m)
    b = 2.4e-3              # Demi-profondeur (m)
    
    # --- 2. MODÉLISATION DU CHAMP Bz (Horton & Moore) ---
    def calculate_Bz(z, x=0, y=0):
        """Modèle analytique des charges de surface pour un plot rectangulaire."""
        res = 0
        # Somme sur les 4 coins du rectangle (alternance de signes)
        for i, xi in enumerate([-a, a]):
            for j, yj in enumerate([-b, b]):
                sign = (-1)**(i + j + 1)
                num = (x - xi) * (y - yj)
                den = z * np.sqrt((x - xi)**2 + (y - yj)**2 + z**2)
                res += sign * np.arctan(num / den)
        return res

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
        s = z0 * np.ones_like(t) # Position de repos
        v = np.sqrt(T / mu)
        for i in range(1, n + 1):
            Bn = (A0 * L**2.5 * np.sqrt(2)) / (i**2 * np.pi**2 * x0 * (L - x0)) * np.sin(i * np.pi * x0 / L)
            s += Bn * np.sin(i * np.pi * x / L) * np.cos(i * 2 * np.pi * f0 * t)
        return s


    def v(n, x, t, A0, L, T, mu, x0):
        """Vitesse de la corde

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
            np.ndarray: Vitesse de la corde
        """
        s = 0
        v = np.sqrt(T / mu)
        for i in range(1, n + 1):
            Bn = (A0 * L**2.5 * np.sqrt(2)) / (i**2 * np.pi**2 * x0 * (L - x0)) * np.sin(i * np.pi * x0 / L)
            An = 2*i*np.pi*f0*np.sin(i * np.pi * x / L)
            s += Bn * An * np.cos(i * 2 * np.pi * f0 * t)
        return s

    # --- 3. GÉNÉRATION DU SIGNAL ---
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Mouvement de la corde (z_s) et vitesse (v_s)
    zs = entree(n=50, x=xm, t=t, A0=A0, L=L, T=T, mu=mu, x0=x0)
    vs = v(n=50, x=xm, t=t, A0=A0, L=L, T=T, mu=mu, x0=x0)
    
    # Calcul du champ et du gradient spatial (dBz/dz)
    dz = 1e-6 # incrément pour la dérivation numérique
    bz_vals = np.array([calculate_Bz(z) for z in zs])
    bz_plus = np.array([calculate_Bz(z + dz) for z in zs])
    dbz_dz = (bz_plus - bz_vals) / dz
    
    # Tension induite e(t) proportional to d(Bz^2)/dt
    # e(t) = -K * 2 * Bz * (dBz/dz) * vs
    induced_voltage = -2 * bz_vals * dbz_dz * vs
    induced_voltage /= np.max(np.abs(induced_voltage)) # Normalisation

    # --- 4. ANALYSE SPECTRALE (0 - 800 Hz) ---
    N = len(t)
    freqs = np.fft.rfftfreq(N, 1/fs)
    spectrum = np.abs(np.fft.rfft(induced_voltage))

    # --- 5. AFFICHAGE DES RÉSULTATS ---
    plt.figure(figsize=(10, 7))
    
    # Graphique Temporel
    plt.subplot(2, 1, 1)
    plt.plot(t[:1000] * 1000, induced_voltage[:1000])
    plt.title("Signal temporel (Modèle de Horton & Moore)")
    plt.xlabel("Temps (ms)")
    plt.ylabel("Tension (u.a.)")
    plt.grid(True, alpha=0.3)
    
    # Graphique Spectral
    plt.subplot(2, 1, 2)
    plt.plot(freqs, spectrum)
    plt.title("Spectre de fréquences (FFT)")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(10, 800) # Focus sur les premières harmoniques
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_guitar_pickup()