from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


def model_factory(model, type, random=42):
    result = None
    if type is "regression":
        if model is "tree":
            result = DecisionTreeRegressor(random_state=random)
    return result


def get_mae(X, y, model):
    predicted_y = model.predict(X)
    return mean_absolute_error(y,predicted_y)

