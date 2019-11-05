import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.sandbox.stats.multicomp as sm

df_mice = pd.read_excel('data/Data_Cortex_Nuclear.xls')
df_mice.fillna(inplace = True)
print(df_mice.head())