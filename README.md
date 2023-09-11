# synthetic_data

Pour l'instant, contient les algorithmes de SDV pour la génération de données synthétiques.
Une interface graphique permet à l'utilisateur de sélectionner son fichier de données, afin d'avoir des données synthétiques qui correspondent.
Une page affiche alors des rapports sur cette génération afin d'évaluer leur qualité.

## Requis
- python
- sdmetrics
- sdv
- csv
- panda
- flask

## Installation 
<code>conda create -n "synthetic"
 conda activate synthetic
 conda install -c pytorch -c conda-forge sdv
 pip install flask
 pip install flask_bootstrap
 pip install panda
 pip install -U kaleido
 pip install sdmetrics</code>


## Lancement
Lancez : 
 
<code>python interfaceGraphiqueEtAlgoGenerationDonnees.py</code>

Puis, suivez les instructions sur la page web locale : http://127.0.0.1:5000
Après que l'algorithme choisi ait généré des données synthétiques, une page s'affiche, avec un rapport d'analyse des données synthétiques, notamment des graphes représentant la distribution des données réelles et des données synthétiques afin de pouvoir comparer ces distributions. 
Le dossier "static/fake_data" contient les données synthétiques générées. 
Le dossier "static/img" contient les différents graphes du rapport d'analyse. 
Le dossier "static/résultats" contient les rapports en .txt concernant ces données synthétiques.


