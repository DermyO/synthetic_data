#This program launches a user interface allowing users to adapt parameters for the generation of synthetic data.
#After users choose the parameters, the program calls the testDifferentalgosFakeData.py algorithm to generate the synthetic data.
#Then, a new page appears showing some reports about the generated data.
#Generated data are keeped in the static/fake_data directory.

# By Oriane Dermy 09/06/2023

#requirement : pythonX, sdmetrics, sdv, csv, panda, flask

import sdv
import csv
import pandas as pd
import warnings
import subprocess
import os
import signal

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

from sdmetrics.reports import utils
from sdmetrics import load_demo
from sdmetrics.reports.single_table import QualityReport

from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot
from sdv.evaluation.single_table import get_column_pair_plot


#function to close the program & flask_server when we close the programm
def handle_sigint(signal, frame):
    shutdown_flask_server()
def shutdown_flask_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is not None:
        func()

app = Flask(__name__)
Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		nameAlgo = request.form.get('nameAlgo')
		dataPath = request.form.get('dataPath')
		dataType = request.form.get('dataType')
		column_name = request.form.get('column_name')
		column_names = request.form.get('column_names')
		# Execution of the python program
		result = execute_code(nameAlgo, dataPath, dataType, column_name, column_names)
		# delete temporary code
		os.remove('temp_code.py')
		# Generate images path
		image_path = 'static/img/column_plot.png' 
		image_path2 = 'static/img/column_pair_plot.png'    
		image_path3 = 'static/img/column_shapes.png'
		return render_template('result.html', result=result, image_path=image_path, image_path2=image_path2, image_path3=image_path3)

	return render_template('index.html')

def execute_code(nameAlgo, dataPath, dataType, column_name, column_names):
	with open('./testDifferentsAlgosFakeData.py', 'r') as f:
		code = f.read()
	print(f"from index, nameAlgo = {nameAlgo}")
	code = code.replace("nameAlgo = 'FAST_ML'", "nameAlgo = '{}'".format(nameAlgo))
	code = code.replace("dataPath = 'default'", "dataPath = '{}'".format(dataPath))
	code = code.replace("dataType = '.csv'", "dataType = '{}'".format(dataType))
	code = code.replace("column_name = 'amenities_fee'", "column_name ='{}'".format(column_name))
	code = code.replace("column_names = ['checkin_date', 'checkout_date']", "column_names={}".format(column_names))
	#print(code)
	
	with open('temp_code.py', 'w') as f:
		f.write(code)

	result = subprocess.check_output(['python', 'temp_code.py']).decode()
	return result

if __name__ == '__main__':
    app.run(debug=True)
