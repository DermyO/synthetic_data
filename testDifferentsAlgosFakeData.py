#requirement : pythonX, sdmetrics, sdv, csv, panda, pip install -U kaleido

import sdv
import csv
import pandas as pd
import warnings

from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata

from sdv.lite import SingleTablePreset
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

from sdv.evaluation.single_table import run_diagnostic
from sdmetrics.multi_table import CardinalityShapeSimilarity
from sdmetrics.column_pairs import CorrelationSimilarity
from sdmetrics.reports import utils
from sdmetrics import load_demo
from sdmetrics.reports.single_table import QualityReport

from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot
from sdv.evaluation.single_table import get_column_pair_plot


def  evaluation(real_data, synthetic_data, metadata, column_name, column_names):
	
	quality_report = evaluate_quality(
		real_data,
		synthetic_data,
		metadata
	)
	
	quality_report.get_visualization('Column Shapes')
	s=f" Quality_report:\n  {quality_report} \n real data:\n {real_data} \n synthetic data: \n {synthetic_data}"
	
	#print(s)
	
	diagnostic_report = run_diagnostic(
		real_data=real_data,
		synthetic_data=synthetic_data,
		metadata=metadata
	)
	
	s+= f"\n Diagnostic report: \n {diagnostic_report.get_results()}"
	print(s)
	
	fig = get_column_plot(
		real_data=real_data,
		synthetic_data=synthetic_data,
		column_name=column_name,
		metadata=metadata
	)
	# Enregistrer la figure dans un fichier d'image
	fig.write_image('static/img/column_plot.png')
	#fig.show()
	
	fig = get_column_pair_plot(
		real_data=real_data,
		synthetic_data=synthetic_data,
		column_names= column_names,
		metadata=metadata
	)
	fig.write_image('static/img/column_pair_plot.png')
	#fig.show()
	
	fig = quality_report.get_visualization(property_name='Column Shapes')
	fig.write_image('static/img/column_shapes.png')

def  evaluationSimple(real_data, synthetic_data, metadata, column_name, column_names):

	quality_report = evaluate_quality(
		real_data,
		synthetic_data,
		metadata
	)

	print(quality_report)

def createSynthetiser(name, metadata):
	if(name=='GaussianCopula'):
		synthesizer = GaussianCopulaSynthesizer(
			metadata, # required
			enforce_min_max_values=True, #controle que données synthétiques gardent max/min des données réelles
			enforce_rounding=False, #controle que les données synthétiques ont même nombre de décimale que données réelles
			#numerical_distributions={ #donne forme de la distribution pour les données numériques
			#	'amenities_fee': 'beta', #norm' 'beta', 'truncnorm', 'uniform', 'gamma' or 'gaussian_kde'
			#	'checkin_date': 'uniform'
			#},
			#locale : liste qui montre type de données qu'on utilise.
			default_distribution='norm' #defaut : beta, sinon les memes autres preuvent etre mises
		)
		#synthesizer.get_parameters()
		#metadata = synthesizer.get_metadata()
	elif(name=='CTGAN'):
		synthesizer = CTGANSynthesizer(
			metadata, # required
			enforce_rounding=False,
			epochs=500, #nombre de fois où on entraine GAN (défaut 300)
			verbose=True #ecris résultat à chaque époch
			#locale, enforce_min_max, CUDA (T/F)
		)
		
	elif(name=='TVAE'):
		synthesizer = TVAESynthesizer(
			metadata, # required
			enforce_min_max_values=True,
			enforce_rounding=False,
			epochs=500 #300 par defaut
			#locales, CUDA
		)
		
	elif(name=='CopulaGAN'):
		synthesizer = CopulaGANSynthesizer(
			metadata, # required
			enforce_min_max_values=True,
			enforce_rounding=False,
			#numerical_distributions={
			#	'amenities_fee': 'beta',
			#	'checkin_date': 'uniform'
			#},
			epochs=500,
			verbose=True
		)
	else:
		synthesizer = SingleTablePreset(
			metadata,
			name=nameAlgo
			#locales=['en_US', 'en_CA', 'fr_FR'] #https://faker.readthedocs.io/en/master/locales.html
		)
		#utilise GaussianCopula avec paramètres fixés : 
		#GaussianCopulaSynthesizer(
		#GaussianCopulaSynthesizer(
		#	enforce_min_max_values=True,
		#	enforce_rounding=True,
		#	default_distribution='norm',
		#)
	return synthesizer

def retrieveDataFromFile(dataPath, dataType):
	if(dataType=='.csv'):
		dataframe = pd.read_csv(dataPath)
		metadata = SingleTableMetadata()
		metadata.detect_from_csv(filepath=dataPath)
		#print("metadata validation:")
		#print(metadata.validate())
		#print("metadata")
		#print(metadata)
		return dataframe, metadata

warnings.filterwarnings("ignore")

nameAlgo = 'FAST_ML'
dataPath = 'default'   # 'data/real_data/data_Distinction.csv'
dataType = '.csv'


column_name = 'amenities_fee'  #'totClicks_108'
column_names = ['checkin_date', 'checkout_date']  #['score_10', 'delay_10']

#recupération données
if(dataPath=='default'):
	real_data, metadata = download_demo(
		modality='single_table',
		dataset_name='fake_hotel_guests'
	)
	column_name = 'amenities_fee'
	column_names=['checkin_date', 'checkout_date']
else:
	real_data, metadata= retrieveDataFromFile(dataPath, dataType)
	#if(column_name ==''):
		#keys = metadata.columns.keys()
		#column_name = '?'
		#column_names = [?,?]
	#print(real_data)

#creation modèle de génération de données
synthesizer = createSynthetiser(nameAlgo, metadata)

print(f"name algo:\n {nameAlgo}")

#apprentissage du modèle
synthesizer.fit(
	data=real_data
)

# génération de données synthétiques
synthetic_data = synthesizer.sample(
	num_rows=500
)

#affichage des données générées
#print(synthetic_data)


#enregistrement des données
dernier_separateur = dataPath.rfind('/')
# Extraire la sous-chaîne après le dernier séparateur
nameFile = dataPath[dernier_separateur + 1:]

print(nameFile)
synthetic_data.to_csv('static/fake_data/fake_'+nameAlgo+'_'+nameFile,index=False)

#évaluation des données
evaluation(real_data, synthetic_data, metadata, column_name, column_names)


#enregistrement du générateur
#synthesizer.save('synthetiser' + nameAlgo + '.pkl')
#récupération du générateur
#synthesizer = SingleTablePreset.load('synthetiser' + nameAlgo + '.pkl')

