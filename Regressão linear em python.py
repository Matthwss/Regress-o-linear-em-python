import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Variaveis a receber valor do usuario
x = list()
y = list()

# Quantidade de dados que o usuário deseja informar
quantidade = int(input("Quantos valores deseja informar? "))

# Entrada dos dados
for c in range(quantidade):
    print(f"\nDados {c + 1}")
    valor_x = float(input("Digite um número para X: "))
    valor_y = float(input("Digite um número para Y: "))
    x.append(valor_x) #add valores as variaveis
    y.append(valor_y)

# Transformando os dados em arrays
X = np.array(x).reshape(-1, 1)
Y = np.array(y)

# Criando o modelo de regressão linear
modelo = LinearRegression()

# Treinando o modelo
modelo.fit(X, Y)

# Pegando os valores da equação
coeficiente = modelo.coef_[0]
intercepto = modelo.intercept_

# Calculando o R²
r2 = modelo.score(X, Y)

print("\n==============================")
print("       RESULTADO")
print("==============================")

print(f"Coeficiente angular: {coeficiente:.2f}")
print(f"Intercepto: {intercepto:.2f}")

print("\nEquação da regressão:")
print(f"Y = {coeficiente:.2f}X + {intercepto:.2f}")

print(f"\nR² = {r2:.2f}")

# Criando os valores previstos
previsao = modelo.predict(X)

# Criando o gráfico
plt.scatter(x, y, label="Valores reais")
plt.plot(x, previsao, label="Regressão linear")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Regressão Linear")
plt.legend()
plt.grid()
plt.show()

# Previsão de um novo valor
novo_x = float(input("\nDigite um novo valor de X para fazer uma previsão: "))

novo_y = modelo.predict([[novo_x]])

print("\n==============================")
print("       PREVISÃO")
print("==============================")

print(f"Para X = {novo_x}, o valor previsto de Y é: {novo_y[0]:.2f}")
