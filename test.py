import pytest
from joblib import load
import pandas as pd

class TestEvaluation:
    @pytest.fixture
    def getData(self):
        data = pd.read_csv('data/test_iris.csv')
        return data

    def test_validation(self, getData):
        assert getData.shape[0] >= 1, f'Atleast one record is expected'
        print('Atleast one record found')

    def test_evaluation(self, getData):
        mod_dt = load("./artifacts/model.joblib")
        X_test = getData[['sepal_length','sepal_width','petal_length','petal_width']]
        prediction=mod_dt.predict(X_test)
        assert prediction[0] == 'versicolor', f'\n Expected versicolor but got {prediction[0]}'
        print(f'Test correctly predicts versicolor')


