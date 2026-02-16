import numpy as np
import matplotlib.pyplot as plt

def lire_fichier(PATH):
    List_temps = []
    List_sig = []
    file_s2 = open(PATH)
    L_lignes = file_s2.readlines()

    for ligne in L_lignes[2:]:
        ligne = ligne.strip().split(',')
        ligne = [x.replace('\n', '') for x in ligne]
        ligne = [x.replace('E', 'e') for x in ligne]
        List_temps.append(float(ligne[0]))
        List_sig.append(float(ligne[1]))
    
    return List_temps, List_sig

def write_file(PATH, List_temps, List_sig):
    with open(PATH, 'w') as file:
        file.write('Time;Signal\n')
        for t, s in zip(List_temps, List_sig):
            file.write(f"{t}\t{s}\n")


def moyenne_mobile(List_temps, List_sig, window_size=100):
    """Filtre par moyenne mobile (moving average).

    Params
    - List_temps: liste des temps
    - List_sig: liste des signaux
    - window_size: taille de la fenêtre (entier).
    """
    sig_array = np.array(List_sig)

    # Protection contre window_size trop grand ou invalide
    if window_size < 1:
        window_size = 1
    window_size = int(window_size)

    # Utiliser une moyenne mobile via convolution (mode 'same' conserve la
    # longueur du signal). On utilise une fenêtre uniforme.
    kernel = np.ones(window_size) / window_size
    filtered_sig = np.convolve(sig_array, kernel, mode='same')

    return filtered_sig


def filtre_passe_bas_1er_ordre(List_temps, List_sig, tau=0.001):
    """Filtre passe-bas du 1er ordre (IIR - Infinite Impulse Response).

    Params
    - List_temps: liste des temps
    - List_sig: liste des signaux
    - tau: constante de temps du filtre (en secondes).
    """
    sig_array = np.array(List_sig)

    # Protection contre tau invalide
    if tau < 0:
        tau = 0.001

    # Calculer l'intervalle de temps moyen (en secondes)
    if len(List_temps) > 1:
        dt = (List_temps[-1] - List_temps[0]) / (len(List_temps) - 1)
    else:
        dt = 1.0

    # Calculer le coefficient du filtre 1er ordre
    # alpha = dt / (tau + dt) pour un filtre passe-bas discretise
    alpha = dt / (tau + dt)

    # Appliquer le filtre de manière causale : y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    filtered_sig = np.zeros_like(sig_array)
    filtered_sig[0] = sig_array[0]
    for i in range(1, len(sig_array)):
        filtered_sig[i] = alpha * sig_array[i] + (1 - alpha) * filtered_sig[i - 1]

    return filtered_sig


def comparer_filtres(PATH_in, PATH_out, window_size=100, tau=0.001):
    """Compare les deux methodes de filtrage sur un seul graphique.

    Params
    - PATH_in: chemin du fichier d'entree (lu par `lire_fichier`)
    - PATH_out: chemin du fichier à ecrire (format tabule) - sauvegarde le filtre passe-bas
    - window_size: taille de la fenêtre pour la moyenne mobile
    - tau: constante de temps pour le filtre passe-bas
    """
    List_temps, List_sig = lire_fichier(PATH_in)
    
    # Appliquer les deux filtres
    filtered_moy = moyenne_mobile(List_temps, List_sig, window_size)
    filtered_pb = filtre_passe_bas_1er_ordre(List_temps, List_sig, tau)

    # Creer le graphique comparatif
    plt.figure(figsize=(6, 5))
    plt.plot(List_temps, List_sig, label='Signal brut', color="#636363", linewidth=1)
    plt.plot(List_temps, filtered_moy, label='Moyenne mobile', linewidth=2, color="#EE2222")
    plt.plot(List_temps, filtered_pb, label='Passe-bas 1er ordre', linewidth=2, color="#0066CC")
    plt.xlabel('Temps [$s$]')
    plt.ylabel('Tension induite dans le micro [$V$]')
    plt.title('Comparaison des methodes de filtrage')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(PATH_out.replace('.csv', '_filter_plot.png'))
    plt.close()
    
    # Sauvegarder le filtre passe-bas (ou les deux si souhaite)
    write_file(PATH_out, List_temps, filtered_pb)


L_path_in = ['Python/data/raw/02-Reponse_meca/Hp1.csv',
            'Python/data/raw/02-Reponse_meca/Hp2.csv',
            'Python/data/raw/02-Reponse_meca/Hp3.csv',
            'Python/data/raw/02-Reponse_meca/Hs1.csv',
            'Python/data/raw/02-Reponse_meca/Hs2.csv',
            'Python/data/raw/02-Reponse_meca/Hs3.csv',
            'Python/data/raw/02-Reponse_meca/S1.csv',
            'Python/data/raw/02-Reponse_meca/S2.csv']

L_path_out = ['Python/data/filter/02-Reponse_meca/Hp1.csv',
            'Python/data/filter/02-Reponse_meca/Hp2.csv',
            'Python/data/filter/02-Reponse_meca/Hp3.csv',
            'Python/data/filter/02-Reponse_meca/Hs1.csv',
            'Python/data/filter/02-Reponse_meca/Hs2.csv',
            'Python/data/filter/02-Reponse_meca/Hs3.csv',
            'Python/data/filter/02-Reponse_meca/S1.csv',
            'Python/data/filter/02-Reponse_meca/S2.csv']

for i in range(len(L_path_in)):
    print(f"Fichier : {L_path_in[i]} : ...")
    comparer_filtres(L_path_in[i], L_path_out[i])
    print(f"Fichier : {L_path_in[i]} : done")