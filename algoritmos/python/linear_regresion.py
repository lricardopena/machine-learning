# Agregamos las librerias que vamos a requerir
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Obtenemos los datos de entrada, en este caso [1,2,6, 13]
X = np.array([[-1, ], [1, ], [2, ], [6, ], [13, ]])

# Calculamos la salida como y = 4.5 * x + 8
y = np.dot(X, np.array([4.5, ])) + 8

# Agregamos un poco de ruido para que no sea una linea perfecta
ruido = np.random.normal(0, 3, len(y))
y += ruido

# Instanciamos el modelo de regresion lineal
regr = linear_model.LinearRegression()

# Entrenamos el modelo utilizando los datos de entrenamiento
regr.fit(X, y)

# Imprimos que el score que tenemos
print("Score {}".format(regr.score(X, y)))

# Imprimimos la pendiente y el bias
print('Pendiente: {0} \nBias: {1}'.format(regr.coef_, regr.intercept_))


# Mostramos la salida y la linea aprendida
plt.scatter(X, y,  color='black')
X_linea = np.linspace(-1, 13, num=20).reshape(-1, 1)
y_linea = regr.predict(X_linea)
plt.plot(X_linea, y_linea, color='blue', linewidth=3)

plt.show()
