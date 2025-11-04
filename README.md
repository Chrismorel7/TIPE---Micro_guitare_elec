# TIPE — Micro guitare électrique

Projet TIPE (Travail d'Initiative Personnelle Encadrée) — Influence du type de micro de guitare électrique sur le son perçu

## Bibliographie Commentée

Contrairement à une guitare dite acoustique, une guitare électrique ne possède pas de caisse de résonnance. Pour produire du son, cette dernière amplifie la vibration de la corde via des micros pour transformer le signal mécanique en signal électrique, qui est ensuite reconvertit en signal acoustique et amplifié par l’amplificateur.

Ces micros de guitare électrique sont en vérité une technologie relativement récente. En effet, malgré une première invention en 1890 par George Breed [1], la première guitare électrique connaissant un succès commercial fut inventée en 1932 par l’entreprise Rickenbacker [2]. Le fonctionnement à l’époque est assez rudimentaire et basé sur un principe connu de la Physique : La loi d’induction de Faraday [3].

Principalement composé d’une bobine de cuivre et d’aimant permanent, le micro vient fournir à la corde un champ magnétique. Lorsque celle-ci va vibrer, elle va engendrer une variation de flux dans la bobine qui va générer une tension électrique. Tension qui sera transmise à l’amplificateur.

En revanche, ces micros (appelé SingleCoil) connaissent vite une limite puisqu’ils détectent également les champs magnétiques ambiant, parasite donc, qui vont détériorer la qualité sonore. C’est pourquoi en 1955 fut inventé un nouveau type de Micro par l’ingénieur Seth Lover : Le Humbucker [3].

En effet, sur scène, le guitariste joue en présence de nombreux appareil électronique destiné à l’éclairage, la sonorisation, les accessoires ou encore à l’alimentation de toute l’installation. Ces divers composants génèrent malgré eux un champ magnétique ambiant (souvent à 50Hz) qui peut s’avérer parasite et produire un son non souhaité par le guitariste. [4]

Composé cette fois ci de deux bobines enroulées dans des sens opposé, le micro Humbucker permet de réduire voire même supprimer ces bruits parasites et ainsi de reproduire le son produit par la corde dans de meilleures conditions. [3]

Mais quels phénomènes physiques interviennent précisément dans l’annulation de bruit parasite ? N’y a-t-il que des avantages à utiliser un micro Humbucker ? Quels sont les limites de cette technologie ?

## Problématique

Comment le type de micro influence la qualité et le timbre du son produit par une guitare électrique ?

## Objectifs du TIPE

- Déterminer la tension induite en entrée du micro en fonction des différents paramètres électro-magnétiques du micro et physique de la guitare sous forme d’équation
- Modéliser l’action du micro sous forme d’un filtre RLC équivalent
- Comparer numériquement les spectres des signaux électriques en sortie des micros SingleCoil Humbucker avec et sans présence d’un champ parasite
- Concevoir et réaliser une maquette de guitare électrique
- Réaliser des mesures en réponse à une excitation standard de la corde en présence ou non d’un champ parasite
- Comparer les écarts entre le modèle théorique et les résultats expérimentaux pour conclure

## Reférence Bibliographique

[1] Matthew Hill. “The hidden history of the electric guitar.” Benoît Navarret, Marc Battier, Philippe Bruguière, Philippe Gonin. ‘’Quand la guitare [s’]électrise!’’, Sorbonne Université Presses, pp.33-61, 2022.

[2] Rickenbacker, “Rickenbacker Legacy”, Rickenbacker, [En ligne]. Disponible sur : https://www.rickenbacker.com/legacy/

[3] S. W. Duncan, "Interview With Seth Lover," Seymour Duncan, [En ligne]. Disponible sur : https://www.seymourduncan.com/blog/latest-updates/seymour-w-duncans-interview-with-seth-lover

[4] Larry, ‘’How to prevent humming and noise in a guitar amp’’, Thomann, [En ligne]. Disponible sur : https://www.thomann.de/blog/en/learn/how-to-prevent-humming-and-noise-in-a-guitar-amp/

[5] Dan Carson. ‘’ Investigation of the Electromagnetic Properties of Electric Guitar Pickups”. University of Illinois at Urbana-Champaign. USA. 2009

[6] P. Perov, W. Johnson, and N. Perova-Mello, "The physics of guitar string vibrations," American Journal of Physics, vol. 84, no. 1, pp. 38–43, Jan. 2016.

## Structure du dépôt

- Attachment : Pièce Jointe et screenshot utile pour la présentation
- Bibliography : Liste des reférences Bibliographiques
- Docs : Documents de Simulation (Logiciels spécifiques)
  - LTSpice : Modélisation éléctrique
  - SolidWorks : Modelisation 3D
- Photo
- Python : Simulation numérique
  - Python/Code/Modele_fem.py — modèle temporel de la corde et calcul des FEM induites.
  - Python/Code/Modele_elec.py — calcul et affichage de la réponse en fréquence (diagramme de Bode) à partir des données.
  - Python/data/Bode.txt — données (fréquence, gain, phase).
  - Python/Exports/ — exports graphiques.
- Tasks : Liste de tache à faire
- Test : Resultats experimentaux
- Views : Présentation
  - MCOT
  - PowerPoint

## Licence

Christophe Morel - Lycée F. Buisson, dit "La Nat"

<!-- Fin du README -->
