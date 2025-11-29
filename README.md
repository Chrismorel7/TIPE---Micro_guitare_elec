# MCOT : Mise en Cohérence des Objectifs du T.I.P.E.

## Titre :  Guitare électrique : Etude de l’influence d u type de micro

## Bibliographie Commentée :
L’apparition de la guitare électrique au XXe siècle a marqué l’histoire de la musique. Née suite à la nécessité d’amplifier le son pour rivaliser avec les percussions et les cuivres dans les grands ensembles, elle a transformé le paysage musical.  
Contrairement à une guitare dite acoustique, qui utilise une caisse de résonnance pour amplifier le son, une guitare électrique est basée uniquement sur un système de transduction électromagnétique pour convertir la vibration mécanique de la corde en signal électrique.
Le dispositif de conversion, le micro, est en vérité une technologie relativement récente. En effet, malgré une première invention en 1890 par George Breed [1], la première guitare électrique connaissant un succès commercial fut inventée en 1932 par l’entreprise Rickenbacker [1]. Le fonctionnement à l’époque est assez rudimentaire et basé sur un principe connu de la Physique : La loi d’induction de Faraday [2]. 
Principalement composé d’une bobine de cuivre et d’aimant permanent, ces derniers viennent plonger les cordes dans un champ magnétique. Lorsque celles-ci vont vibrer [3], elles vont engendrer une variation de flux dans la bobine qui va générer une force électromotrice (FEM) [2]. C’est ce signal électrique qui  est ensuite transmis à l’amplificateur et convertit en son.
En revanche, ces micros (appelé SingleCoil) connaissent vite une limite puisqu’ils détectent également les champs magnétiques ambiant, parasites donc, qui vont détériorer la qualité sonore. En effet, sur scène, le guitariste joue en présence de nombreux appareils électroniques destinés à l’éclairage, la sonorisation, les accessoires ou encore à l’alimentation de toute l’installation. Ces divers composants génèrent malgré eux un champ magnétique ambiant (souvent à 50Hz) qui peut s’avérer parasite et produire un son non souhaité par le guitariste. [4]
Dès lors, il vient la question suivante : Comment contrer ces effets parasites sans perdre en qualité ?
En 1955 fut inventé un nouveau type de micro par l’ingénieur Seth Lover : Le Humbucker [4]. Ce micro, censé réduire considérablement le bruit parasite, voire parfois l’annuler, permet ainsi de reproduire le son généré par la corde dans de meilleures conditions.
L'architecture du Humbucker est ingénieuse : il est composé de deux bobines en série, exploitant deux principes d'inversion pour obtenir une annulation sélective :
-	Opposition de phase : Les bobines sont câblées dans des sens opposées. Tout champ induit alors une FEM opposée dans chaque bobine, conduisant à une soustraction et une annulation du bruit. 
-	Polarité inversée : Les aimants sous chaque corde ont des polarités opposées, cette inversion magnétique compense l’inversion du bobinage uniquement dans le cas où le signal est produit par la corde, assurant l’addition du signal.
Cependant, cette architecture double a un impact direct sur le timbre du son. Les guitaristes, au départ séduits par cette technologie, déplorent une perte des hautes fréquences et un son plus rond par rapport au micro SingleCoil [4]. 
Ce phénomène peut être expliqué par le comportement de type Passe-bas des micros. La mise en série des bobines augmente l’inductance totale du système. La fréquence de coupure s’abaisse et entraine alors une diminution des harmoniques élevées. [2]
Ainsi pour le guitariste et le luthier, il s’agit de trouver le parfait compromis entre perte des hautes fréquence et annulation des bruits parasites.
Je m’intéresserai alors à quantifier ces phénomènes de façon précise afin de pouvoir déterminer par quelles composantes du micro ils sont impactés. Enfin, il s’agira pour moi de déterminer de taux de rejection de mode commun (CMRR) du micro Humbucker afin de statuer sur son efficacité.

## Problématique :
Comment le type de micro influence la qualité (pureté) et le timbre du son produit par une guitare électrique ?

## Objectifs :
•	Déterminer la FEM induite en fonction des harmoniques de vibration de la corde pour quantifier l’influence sur le timbre
•	Etablir la fonction de transfert du circuit RLC équivalent des micros pour quantifier l’influence sur le timbre
•	Théoriser la FEM résiduelle du Humbucker sous un champ parasite non uniforme pour quantifier sur la qualité
•	Concevoir une maquette permettant de valider et/ou critiquer cette théorie
•	Déterminer le taux de rejection de mode commun du micro Humbucker
•	Conclure sur les avantages et les inconvénients du micro Humbucker

## Référence Bibliographique :
[1] Matthew W. Hill, "The hidden history of the electric guitar", in Quand la guitare [s']électrise !, Benoît Navarret et al. (dir.), Sorbonne Université Presses, pp. 33-61, 2022.
[2] Dan Carson, "Investigation of the Electromagnetic Properties of Electric Guitar Pickups", University of Illinois at Urbana-Champaign, Physics Senior Thesis, 2009.
[3] P. Perov, W. Johnson et N. Perova-Mello, "The physics of guitar string vibrations", American Journal of Physics, vol. 84, no. 1, pp. 38–43, 2016.
[4] S. W. Duncan, "Interview With Seth Lover", Seymour Duncan.  [En ligne]. Disponible sur : 
https://www.seymourduncan.com/blog/latest-updates/seymour-w-duncans-interview-with-seth-lover
[5] A. Novak, B. Lihoreau, E. Brasseur, P. Lotton et L. Simon, "Guitar Pickup: Measurement of its Two-Dimensional Nonlinear Function", Journal of the Audio Engineering Society, vol. 68, no. 3, pp. 184-192, 2020.
[6] R. Giampiccolo, A. Bernardini et A. Sarti, "Virtualization of Guitar Pickups Through Circuit Inversion", IEEE Signal Processing Letters, vol. 30, pp. 458-462, 2023.

## Structure du dépot
•	soon...
