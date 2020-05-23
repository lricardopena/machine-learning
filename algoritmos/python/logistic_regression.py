# Importamos las librerias que vamos a necesitar
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Obtenemos los datos de entrada
X, y = load_iris(return_X_y=True)

# Entrenamos el modelo de regresion logistica
clf = LogisticRegression(random_state=0).fit(X, y)

# Predecimos para las muestras 3 y 4
print(clf.predict(X[2:4, :]))

# Imprimimos las probabilidades para cada clase en este caso 3 clases de las muestras 3 y 4
print(clf.predict_proba(X[2:4, :]))
# Mostramos el score
print(clf.score(X, y))


