# TIPE — Micro guitare électrique

Projet TIPE (Travail d'Initiative Personnelle Encadrée) — étude comparée des micros guitare : simple bobinage vs humbucker.

## Objectif

Étant étudiant en CPGE, filière PT, ce TIPE a pour objectifs :

- Modéliser un phénomène physique lié aux micros de guitare.
- Réaliser une expérience ou une simulation permettant de valider un modèle théorique.
- Présenter un rapport clair et concis.

## Contexte et principe de fonctionnement (résumé)

Un micro de guitare se compose principalement d'aimants permanents et d'un bobinage. La corde métallique, magnétisée par l'aimant, modifie le flux magnétique lorsque'elle vibre au‑dessus du bobinage. Cette variation de flux induit une force électromotrice (FEM) dans la bobine, qui est ensuite amplifiée et convertie en signal sonore.

Le humbucker (double bobinage) combine deux spires avec polarité et sens d'enroulement opposés pour :

- Réduire les parasites électromagnétiques par annulation de phase.
- Augmenter le signal utile dans certaines configurations (effet de somme des contributions de corde selon l'architecture).

## Structure du dépôt

- Scripts de simulation :
  - Python/Code/Modele_fem.py — modèle temporel de la corde et calcul des FEM induites.
  - Python/Code/Modele_elec.py — calcul et affichage de la réponse en fréquence (diagramme de Bode) à partir des données.
  - Python/data/Bode.txt — données (fréquence, gain, phase).
  - Python/Exports/ — exports graphiques.
- Documents de présentation : Views/ (PDF/PPTX).
- Plan de travail : Tasks/tasks.todo

## Utilisation rapide

Prérequis : Python 3.x, numpy, scipy, matplotlib (installer via pip si nécessaire).

Exemples :

- Lancer la simulation temporelle :
  python Python/Code/Modele_fem.py
- Générer la réponse en fréquence :
  python Python/Code/Modele_elec.py Python/data/Bode.txt

(Consulter les entêtes des scripts pour les options et paramètres.)

## Méthodologie

- Modélisation analytique / numérique du mouvement de la corde.
- Calcul des FEM induites par variation de flux magnétique (loi de Faraday).
- Simulation de la chaîne électrique micro → ampli et analyse en fréquence.
- Comparaison des réponses et discussion des avantages du humbucker.

## Résultats

Rédaction et synthèse des résultats de simulation et expérimentaux en cours. Les graphiques exportés sont dans Python/Exports/.

## Conclusion

À rédiger (synthèse des observations, validation du modèle, limites et perspectives expérimentales).

## Contributions et contact

Contributions bienvenues via pull request. Pour toute question : voir l'entête du dépôt ou contact auteur dans le rapport.

## Licence

Ajouter une licence si nécessaire.

<!-- Fin du README -->
