import numpy as np
import logging
import matplotlib.pyplot as plt

path = "Python/raw data/SingleCoil_corde_78,5.csv"
try:
    with open(path) as doc:
        L_lignes = doc.readlines()
    Test_temps_h = []
    Test_h = []
    for i in range(35090, len(L_lignes)):
        values = L_lignes[i].strip().split(";")
        values = [v.replace(',', '.') for v in values]
        values = [float(v.replace("\n", "")) for v in values]
        Test_temps_h.append(values[0])
        Test_h.append(values[1])
    Test_temps_h = np.array(Test_temps_h)
    Test_temps_h -= Test_temps_h[0]
    Test_temps_h_fft = np.array(Test_temps_h)
    Test_temps_h = np.array(Test_temps_h)
    Test_h_fft = np.array(Test_h)
    Test_h = - np.array(Test_h)
    print("Donnees de test Humbucker corde seule lues.")
except Exception as e:
    logging.error(f"Erreur lors de la lecture des donnees de test Humbucker corde seule : {e}")

plt.plot(Test_temps_h, Test_h)
plt.show()