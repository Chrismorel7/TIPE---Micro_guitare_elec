import numpy as np
import matplotlib.pyplot as plt

def lire_fichier(PATH):
    List_temps = []
    List_sig = []
    file_s2 = open(PATH)
    L_lignes = file_s2.readlines()

    for ligne in L_lignes[1:]:
        ligne = ligne.strip().split(';')
        ligne = [x.replace('\n', '') for x in ligne]
        ligne = [x.replace(',', '.') for x in ligne]
        List_temps.append(float(ligne[0]))
        List_sig.append(float(ligne[1]))
    
    return List_temps, List_sig

def write_file(PATH, List_temps, List_sig):
    with open(PATH, 'w') as file:
        file.write('Time;Signal\n')
        for t, s in zip(List_temps, List_sig):
            file.write(f"{t}\t{s}\n")


def passebas(PATH_in, PATH_out, window_size=20):
    """Filtre passe-bas par moyenne mobile (moving average).

    Params
    - PATH_in: chemin du fichier d'entrée (lu par `lire_fichier`)
    - PATH_out: chemin du fichier à écrire (format tabulé)
    - window_size: taille de la fenêtre (entier). Plus elle est grande, plus le
    filtrage est agressif (meilleure réduction du bruit haute fréquence).
    """
    List_temps, List_sig = lire_fichier(PATH_in)
    sig_array = np.array(List_sig)

    # Protection contre window_size trop grand ou invalide
    if window_size < 1:
        window_size = 1
    window_size = int(window_size)

    # Utiliser une moyenne mobile via convolution (mode 'same' conserve la
    # longueur du signal). On utilise une fenêtre uniforme.
    kernel = np.ones(window_size) / window_size
    filtered_sig = np.convolve(sig_array, kernel, mode='same')

    # Pour les bords, la convolution 'same' peut introduire un léger biais ;
    # si on veut une version causale on ferait une moyenne cumulée. Ici on garde
    # 'same' pour rester simple et rapide.

    plt.plot(List_temps, List_sig, label='Signal brut')
    plt.plot(List_temps, filtered_sig, label='Signal filtré', linewidth=2)
    plt.xlabel('Temps (s)')
    plt.ylabel('Signal')
    plt.title('Filtrage passe-bas par moyenne mobile')
    plt.legend()
    plt.savefig(PATH_out.replace('.csv', '_filter_plot.png'))
    plt.close()
    write_file(PATH_out, List_temps, filtered_sig)

L_path_in = ['Python/raw data/SingleCoil_par_2V.csv',
            'Python/raw data/SingleCoil_par_4V.csv',
            'Python/raw data/SingleCoil_par_6V.csv',
            'Python/raw data/SingleCoil_par_8V.csv',
            'Python/raw data/SingleCoil_par_10V.csv',
            'Python/raw data/SingleCoil_par_12,5V.csv',
            'Python/raw data/SingleCoil_par_15V.csv',
            'Python/raw data/SingleCoil_par_17,5V.csv',
            'Python/raw data/SingleCoil_par_20V.csv',
            
            'Python/raw data/Humbucker_par_2V.csv',
            'Python/raw data/Humbucker_par_4V.csv',
            'Python/raw data/Humbucker_par_6V.csv',
            'Python/raw data/Humbucker_par_8V.csv',
            'Python/raw data/Humbucker_par_10V.csv',
            'Python/raw data/Humbucker_par_12,5V.csv',
            'Python/raw data/Humbucker_par_15V.csv',
            'Python/raw data/Humbucker_par_17,5V.csv',
            'Python/raw data/Humbucker_par_20V.csv']

L_path_out = ['Python/filtered data/SingleCoil_par_2V.csv',
            'Python/filtered data/SingleCoil_par_4V.csv',
            'Python/filtered data/SingleCoil_par_6V.csv',
            'Python/filtered data/SingleCoil_par_8V.csv',
            'Python/filtered data/SingleCoil_par_10V.csv',
            'Python/filtered data/SingleCoil_par_12,5V.csv',
            'Python/filtered data/SingleCoil_par_15V.csv',
            'Python/filtered data/SingleCoil_par_17,5V.csv',
            'Python/filtered data/SingleCoil_par_20V.csv',
            
            'Python/filtered data/Humbucker_par_2V.csv',
            'Python/filtered data/Humbucker_par_4V.csv',
            'Python/filtered data/Humbucker_par_6V.csv',
            'Python/filtered data/Humbucker_par_8V.csv',
            'Python/filtered data/Humbucker_par_10V.csv',
            'Python/filtered data/Humbucker_par_12,5V.csv',
            'Python/filtered data/Humbucker_par_15V.csv',
            'Python/filtered data/Humbucker_par_17,5V.csv',
            'Python/filtered data/Humbucker_par_20V.csv']

for i in range(len(L_path_in)):
    passebas(L_path_in[i], L_path_out[i], window_size=20)