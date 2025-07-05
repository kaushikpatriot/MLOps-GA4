import joblib
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt 
import mlflow
from mlflow.models import infer_signature
from hyperopt import tpe, space_eval, hp, fmin, Trials, STATUS_OK


##Training the model
data = pd.read_csv('data/iris.csv')

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

mlflow.set_tracking_uri(uri="http://34.123.21.120:8100")

mlflow.set_experiment("MLFlow Iris Classification Experiments")

signature = infer_signature(X_train, y_train)

def train(max_depth):
    mod_dt = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    mod_dt.fit(X_train,y_train)
    prediction=mod_dt.predict(X_test)
    accuracy = metrics.accuracy_score(prediction,y_test)
    print('The accuracy of the Decision Tree is',"{:.3f}".format(accuracy))
    return {
        'model': mod_dt,
        'accuracy': accuracy,
        'loss': 1-accuracy
    }
    # disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, prediction)
    # disp.plot()
    # plt.title("Confusion Matrix for Iris classifier")
    # plt.savefig('./plot.png')

    ## Logging the metrics and params in MLFlow

    # Start an MLflow run

def objective(params):
    with mlflow.start_run(nested = True):
        # Log the hyperparameters

        mlflow.log_params(params)

        result = train(
            max_depth=params["max_depth"]
        )

        # Log the loss metric
        mlflow.log_metric("accuracy", result["accuracy"])

        # Log the model, which inherits the parameters and metric
        model_info = mlflow.sklearn.log_model(
            sk_model=result['model'],
            name="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )

        # Set a tag that we can use to remind ourselves what this model was for
        mlflow.set_logged_model_tags(
            model_info.model_id, {"Training Info": "Basic DT model for iris data"}
        )
        result['status'] = STATUS_OK
        result['model_info'] = model_info
        return result

def test(model_info):
    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(X_test)
    iris_feature_names = datasets.load_iris().feature_names
    result = pd.DataFrame(X_test)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions
    print(result[:4])


search_space = {
    "max_depth": hp.choice("max_depth", [3,4,5]),
}

with mlflow.start_run(run_name="hyperparameter-sweep"):

    # Run optimization
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=trials,
        verbose=True,
    )

    # Find and log best results
    best_trial = min(trials.results, key=lambda x: x["loss"])
    best_accuracy = best_trial["accuracy"]
    best_model = best_trial["model_info"]

    # Log optimization results
    mlflow.log_params(
        {
            "best_max_depth": best_trial["model"].max_depth
        }
    )
    mlflow.log_metrics(
        {
            "best_val_accuracy": best_accuracy,
            "total_trials": len(trials.trials),
            "optimization_completed": 1,
        }
    )


test(best_model)