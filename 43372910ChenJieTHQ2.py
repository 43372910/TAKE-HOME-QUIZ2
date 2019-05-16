#QUESTION2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#part a
data = {'Output,Q': [80,150,135,165,95,130,110],
        'Labour,L': [60,100,100,120,70,90,80],
        'Capital,K': [50,100,80,100,60,80,70]}
df = pd.DataFrame(data, index = ['B','C','D','E','F','G','H'])
print(df)
#calculate mean,variance,standard deviation and median values
df.mean()
df.var()
df.std()
df.median()

#part b
import seaborn as sns
x1 = df['Output,Q']
x2 = df['Labour,L']
#plot histogram for Q
ax1 = sns.distplot(x1)
ax1.set_title('histrogram of Q')
ax1.set_ylabel('density')
#plot histogram for L
ax2=sns.distplot(x2)
ax2.set_title('histrogram of L')
ax2.set_ylabel('density')

#part c
log_df = np.log(df)
print(log_df)

#part d
x = log_df[['Labour,L','Capital,K']]
y = log_df['Output,Q']
import statsmodels.api as sm
X = sm.add_constant(x)
model1= sm.OLS(y,X).fit()
predictions = model1.predict(X)
model1.summary()
0.5484+0.5087

#part e
residual = y-predictions
index = ['B','C','D','E','F','G','H']
residualplot = sns.scatterplot(x = index, y=residual)
residualplot.set_title('residual plot of model')
residualplot.set_ylabel('residual')
residualplot.set_xlabel('countries')
plt.axhline(0,color='red',ls='dotted')

#part f
print('Adjusted R^2:', model1.rsquared_adj)
