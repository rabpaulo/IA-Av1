import numpy as np
import plotly.express as px
class LinearRegression: #criando a classe
    def __init__(self,x, y): #construtor
        self.x = np.array(x) #dados de entrada
        self.y = np.array(y) #dados de saida
        self.b0 = None #intercepto
        self.b1 = None #coeficiente angular

    def fit(self): #treinamento, descobrir os parametros(b0, b1)
        xbar = np.mean(self.x)
        ybar = np.mean(self.y)
        self.b1 = np.sum((self.x - xbar) * (self.y - ybar))/ np.sum((self.x - xbar) **2) #calculo do coeficiente angular
        self.b0 = ybar - self.b1 * xbar #calculo do intercepto
        return self

    def predict(self, x_new): #predicao
            return self.b0 + self.b1 * np.array(x_new)

    def summary(self):
        print(f"Modelo: y = {self.b0} + {self.b1} * x")
        print(f"Intercepto = {self.b0}")
        print(f"Coeficiente angular = {self.b1}")

dados = np.loadtxt(r"./qb.csv", delimiter=",", skiprows=1, usecols=(2, 3))

X = dados[:,0]
Y = dados[:,1]

modelo = LinearRegression(X, Y)
# ------
# Item A
# ------
modelo.fit()
modelo.summary()
print("---")

# ------
# Item B
# ------
estimativaB = modelo.predict(7.5)
print(f"Estimativa da pontuação para 7.5 jardas: {estimativaB:.2f}")
print("---")

# ------
# Item C
# ------
itemc = -modelo.b1
print(f"A mudança na pontuação para uma diminuição de 1 jarda é de: {itemc}")
print("---")

# ------
# Item D
# ------
estimativaD = modelo.predict(7.21)
print(f"Estimativa da pontuação para 7.21 jardas: {estimativaD:.2f}")

indice = np.where(X == 7.21)[0]
print(indice)
if len(indice) > 0:
    posicao = indice[0] 
    print(posicao)
    # Pega o valor real correspondente no array Y
    y_real = Y[posicao] 
    residuo = y_real - estimativaD
    
    print(f"Valor real observado para 7.21 jardas: {y_real}")
    print(f"Resíduo correspondente (Real - Estimado): {residuo:.2f}")

print("---")

# ------
# Item D
# ------
# 1.
valoresAjustados = np.array([modelo.predict(item) for item in X])
# 2.
residuos = np.array(Y - valoresAjustados)
# 3.
print("Observado | Ajustado | Residuo")
for y_obs, y_ajust, res in zip(Y, valoresAjustados, residuos):
    print(f"{y_obs:.2f} | {y_ajust:.2f} | {res:.2f}")
