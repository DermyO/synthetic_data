"""eval_only.py
	13/06/23
	Oriane Dermy
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization


def eval_only (real_data, synthetic_data):  
  print("bien dans eval_only")
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  # 1. Discriminative Score
  discriminative_score = list()
  for _ in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(real_data, generated_data)
    discriminative_score.append(temp_disc)
      
  metric_results['discriminative'] = np.mean(discriminative_score)
      
  # 2. Predictive score
  predictive_score = list()
  for tt in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(real_data, generated_data)
    predictive_score.append(temp_pred)   
      
  metric_results['predictive'] = np.mean(predictive_score)     
          
  # 3. Visualization (PCA and tSNE)
  visualization(real_data, generated_data, 'pca')
  visualization(real_data, generated_data, 'tsne')
  
  ## Print discriminative and predictive scores
  print(metric_results)

  return metric_results
