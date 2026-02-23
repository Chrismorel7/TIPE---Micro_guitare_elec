# Guitare electrique : Etude de l’influence du type de micro

## Bibliographie Commentee :

L’apparition de la guitare electrique au XXe siècle a marque l’histoire de la musique. Nee suite à la necessite d’amplifier le son pour rivaliser avec les percussions et les cuivres dans les grands ensembles, elle a transforme le paysage musical.

Contrairement à une guitare dite acoustique, qui utilise une caisse de resonnance pour amplifier le son, une guitare electrique est basee uniquement sur un système de transduction electromagnetique pour convertir la vibration mecanique de la corde en signal electrique.

Le dispositif de conversion, le micro, est en verite une technologie relativement recente. En effet, malgre une première invention en 1890 par George Breed [1], la première guitare electrique connaissant un succès commercial fut inventee en 1932 par l’entreprise Rickenbacker [1]. Le fonctionnement à l’epoque est assez rudimentaire et base sur un principe connu de la Physique : La loi d’induction de Faraday [2].

Principalement compose d’une bobine de cuivre et d’aimant permanent, ces derniers viennent plonger les cordes dans un champ magnetique. Lorsque celles-ci vont vibrer [3], elles vont engendrer une variation de flux dans la bobine qui va generer une force electromotrice (FEM) [2]. C’est ce signal electrique qui est ensuite transmis à l’amplificateur et convertit en son.

En revanche, ces micros (appele SingleCoil) connaissent vite une limite puisqu’ils detectent egalement les champs magnetiques ambiant, parasites donc, qui vont deteriorer la qualite sonore. En effet, sur scène, le guitariste joue en presence de nombreux appareils electroniques destines à l’eclairage, la sonorisation, les accessoires ou encore à l’alimentation de toute l’installation. Ces divers composants genèrent malgre eux un champ magnetique ambiant (souvent à 50Hz) qui peut s’averer parasite et produire un son non souhaite par le guitariste. [4]

Dès lors, il vient la question suivante : Comment contrer ces effets parasites sans perdre en qualite ?

En 1955 fut invente un nouveau type de micro par l’ingenieur Seth Lover : Le Humbucker [4]. Ce micro, cense reduire considerablement le bruit parasite, voire parfois l’annuler, permet ainsi de reproduire le son genere par la corde dans de meilleures conditions.

L'architecture du Humbucker est ingenieuse : il est compose de deux bobines en serie, exploitant deux principes d'inversion pour obtenir une annulation selective :

- Opposition de phase : Les bobines sont câblees dans des sens opposees. Tout champ induit alors une FEM opposee dans chaque bobine, conduisant à une soustraction et une annulation du bruit.

- Polarite inversee : Les aimants sous chaque corde ont des polarites opposees, cette inversion magnetique compense l’inversion du bobinage uniquement dans le cas où le signal est produit par la corde, assurant l’addition du signal.

Cependant, cette architecture double a unimpact direct sur le timbre du son. Les guitaristes, au depart seduits par cette technologie, deplorent une perte des hautes frequences et un son plus rond par rapport au micro SingleCoil [4].

Ce phenomène peut être explique par le comportement de type Passe-bas des micros. La mise en serie des bobines augmente l’inductance totale du système. La frequence de coupure s’abaisse et entraine alors une diminution des harmoniques elevees. [2]

Ainsi pour le guitariste et le luthier, il s’agit de trouver le parfait compromis entre perte des hautes frequence et annulation des bruits parasites.

Je m’interesserai alors à quantifier ces phenomènes de façon precise afin de pouvoir determiner par quelles composantes du micro ils sont impactes. Enfin, il s’agira pour moi de determiner de taux de rejection de mode commun (CMRR) du micro Humbucker afin de statuer sur son efficacite.

## Problematique :

Comment le type de micro influence la qualite (purete) et le timbre du son produit par une guitare electrique ?

## Objectifs :

• Determiner la FEM induite en fonction des harmoniques de vibration de la corde pour quantifier l’influence sur le timbre

• Etablir la fonction de transfert du circuit RLC equivalent des micros pour quantifier l’influence sur le timbre

• Theoriser la FEM residuelle du Humbucker sous un champ parasite non uniforme pour quantifier sur la qualite

• Concevoir une maquette permettant de valider et/ou critiquer cette theorie

• Determiner le taux de rejection de mode commun du micro Humbucker

• Conclure sur les avantages et les inconvenients du micro Humbucker

## Reference Bibliographique :

[1] Matthew W. Hill, "The hidden history of the electric guitar", in Quand la guitare [s']electrise !, Benoît Navarret et al. (dir.), Sorbonne Universite Presses, pp. 33-61, 2022.

[2] Dan Carson, "Investigation of the Electromagnetic Properties of Electric Guitar Pickups", University of Illinois at Urbana-Champaign, Physics Senior Thesis, 2009.

[3] P. Perov, W. Johnson et N. Perova-Mello, "The physics of guitar string vibrations", American Journal of Physics, vol. 84, no. 1, pp. 38–43, 2016.

[4] S. W. Duncan, "Interview With Seth Lover", Seymour Duncan. [En ligne]. Disponible sur :
https://www.seymourduncan.com/blog/latest-updates/seymour-w-duncans-interview-with-seth-lover

[5] A. Novak, B. Lihoreau, E. Brasseur, P. Lotton et L. Simon, "Guitar Pickup: Measurement of its Two-Dimensional Nonlinear Function", Journal of the Audio Engineering Society, vol. 68, no. 3, pp. 184-192, 2020.

[6] R. Giampiccolo, A. Bernardini et A. Sarti, "Virtualization of Guitar Pickups Through Circuit Inversion", IEEE Signal Processing Letters, vol. 30, pp. 458-462, 2023.

## Structure du depot

• .vscode : Dossier de configuration

• Attachment : Captures d'ecrans et schema pour integrer a la presentation

• Bibliography : Liste des sources (cf. Reference Bibliographique)

• Docs : Documents de modelisation (LTSpice, SWorks)

• Python : Fichier de modelisation, data experimentales (csv)

• Tasks : Liste des taches

• Test : Compte rendu et photo expérimentale

• Views : Fichier de presentation

## License et Credits

Etudiant : Christophe Morel

Filière : CPGE - PT

Etablissement : Lycee F. Buisson, Voiron

Annee : 2024 - 2026
