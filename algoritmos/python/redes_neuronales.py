# Agregamos las librerias que vamos a requerir
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_regression

# ========== Random Forest en clasificacion ===================
# Obtenemos los datos de entrada, en este caso el iris dataset https://archive.ics.uci.edu/ml/datasets/iris
X, y = load_iris(return_X_y=True)

# Instanciamos el modelo de random Forest
ann_model = MLPClassifier()

# Entrenamos el modelo
ann_model = ann_model.fit(X, y)

# Predecimos los 2 primeros datos del iris
print(ann_model.predict(X[:2, :]))


# ========== Random Forest en Regresion ===================
# Obtenemos los datos de entrada, en este caso el iris dataset https://archive.ics.uci.edu/ml/datasets/iris
X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

# Instanciamos el modelo de random Forest
ann_model = MLPRegressor()

# Entrenamos el modelo
ann_model = ann_model.fit(X, y)

# Predecimos los 2 primeros datos del iris
print("Real: {0}\nPrediccion:{1}".format(y[:2], ann_model.predict(X[:2])))