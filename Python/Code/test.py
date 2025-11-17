import matplotlib.pyplot as plt

doc= open("Test/17-11-2025/Test final/Humbucker parasite + corde.txt")
L_lignes = doc.readlines()
Test_temps_tot = []
Test_h_tot = []
for i in range(1, len(L_lignes)):
    values = L_lignes[i].strip().split("\t")
    values = [float(v.replace("\n", "")) for v in values]
    Test_temps_tot.append(values[0])
    Test_h_tot.append(values[1])

plt.plot(Test_temps_tot, Test_h_tot, "+", label="Donnees de test [V]")
plt.show()