# synthetic_data

Pour l'instant, contient les algorithmes de SDV pour la génération de données synthétique.
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
