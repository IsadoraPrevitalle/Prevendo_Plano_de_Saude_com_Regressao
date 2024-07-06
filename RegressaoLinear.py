import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('plano_saude.csv')
print(df)

# Organizar as idades no x plano saude
X = df.iloc[:,0].values #idade
print(X) 

y = df.iloc[:,1].values #custo do plano de saude
print(y)

# Aplicar a correlação
print(np.corrcoef(X,y))

#Colocar em formato de matriz usando o reshape
#Necessário devido a forma como o algoritmo foi criado
X = X.reshape(-1,1)
print(X)

regressor = LinearRegression()

# trinamento para encontar b0 e os coeficientes para os atributos
regressor.fit(X, y)

# mostrando b0 - constante
print(regressor.intercept_)

# mostrando b1 - coeficiente
print(regressor.coef_)

previsao = regressor.predict(X)
print(previsao)

#transformando em vetor para colocar no gráfico

print(X.ravel())

#Testando a equação = b0+b1*40(Idade)
print(regressor.intercept_ + regressor.coef_ * 40)

print(regressor.predict([[40]])) #dois colchetes devido ao formato da matriz

#Score do quão foi bom o algoritmo
print(regressor.score(X,y))

grafico = px.scatter(x=X.ravel(), y=y)
grafico.add_scatter(x=X.ravel(), y=previsao, name='Regressão')
grafico.show()

