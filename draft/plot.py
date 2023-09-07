import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('eval.csv')

data.plot(kind='bar',x='dataset')
plt.show()

