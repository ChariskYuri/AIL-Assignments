import numpy as np
import pandas as pd

df = pd.read_csv('Preprocessed SVM data.csv', usecols=['15'], index_col=False)
print(df.value_counts())
