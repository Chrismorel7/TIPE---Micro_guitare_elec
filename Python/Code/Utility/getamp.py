import numpy as np
import matplotlib.pyplot as plt

def write_file(PATH, List_temps, List_sig):
    with open(PATH, 'w') as file:
        file.write('Time;Signal\n')
        for t, s in zip(List_temps, List_sig):
            file.write(f"{t}\t{s}\n")

def get_amplitude(PATH):
    List_temps = []
    List_sig = []
    file = open(PATH)
    
    L_lignes = file.readlines()
    for ligne in L_lignes[1:70000]:
        ligne = ligne.strip().split('\t')
        ligne = [x.replace('\n', '') for x in ligne]
        ligne = [x.replace(',', '.') for x in ligne]
        List_temps.append(float(ligne[0]))
        List_sig.append(float(ligne[1]))
    
    amplitude = max(List_sig) - min(List_sig)
    return amplitude

L_path = ['Python/filtered data/SingleCoil_par_2V.csv',
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

L_labels = ['Single Coil 2V', 'Single Coil 4V', 'Single Coil 6V', 'Single Coil 8V',
            'Single Coil 10V', 'Single Coil 12.5V', 'Single Coil 15V', 'Single Coil 17.5V',
            'Single Coil 20V',
            'Humbucker 2V', 'Humbucker 4V', 'Humbucker 6V', 'Humbucker 8V',
            'Humbucker 10V', 'Humbucker 12.5V', 'Humbucker 15V', 'Humbucker 17.5V',
            'Humbucker 20V']

for i in range(len(L_path)):
    amp = get_amplitude(L_path[i])
    write_file('Python/Code/amplitudes.csv', L_labels[:i+1], [get_amplitude(p) for p in L_path[:i+1]])