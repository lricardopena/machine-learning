# Agregamos las librerias que vamos a requerir
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_regression
from sklearn import tree

# ========== Arboles de decision en clasificacion ===================
# Obtenemos los datos de entrada, en este caso el iris dataset https://archive.ics.uci.edu/ml/datasets/iris
X, y = load_iris(return_X_y=True)

# Instanciamos el modelo de decision Tree
decision_tree_model = tree.DecisionTreeClassifier()

# Entrenamos el modelo
decision_tree_model = decision_tree_model.fit(X, y)

# Mostramos el modelo
tree.plot_tree(decision_tree_model, filled=True)
plt.show()
# Predecimos los 2 primeros datos del iris
print(decision_tree_model.predict(X[:2, :]))

# ========== Arboles de Decision en Regresion ===================
# Obtenemos los datos de entrada, en este caso el iris dataset https://archive.ics.uci.edu/ml/datasets/iris
X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

# Instanciamos el modelo de random Forest
decision_tree_model = tree.DecisionTreeRegressor(max_depth=6, random_state=0)

# Entrenamos el modelo
decision_tree_model = decision_tree_model.fit(X, y)

# Mostramos el modelo
tree.plot_tree(decision_tree_model, filled=True)
plt.show()

# Predecimos los 2 primeros datos del iris
print("Real: {0}\nPrediccion:{1}".format(y[:2], decision_tree_model.predict(X[:2])))
