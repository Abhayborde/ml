import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data =pd.read_csv("data.csv")
print(data.head())
print(data.tail())
clean=data.dropna()
print(clean.describe())

plt.figure(figsize=(10,6))
sns.histplot(clean['Age'] ,kde="True")
plt.xlabel('Age')
plt.ylabel('Frequence')
plt.show()
t,p=stats.ttest_1samp(clean['Age'],0)
print(t)
print(p)