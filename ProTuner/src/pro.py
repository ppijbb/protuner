import pandas as pd

import eli5
from eli5.sklearn import PermutationImportance
import shap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class tuner:
    def __init__(self, data,target):
        self.X =  data.drop(columns=[target]).copy()
        self.y = data.loc[:, target].values
        x_train, x_test, self.y_train, self.y_test \
            = train_test_split(self.X, self.y, stratify=self.y, test_size=0.222, random_state=42)

        scaler = MinMaxScaler()
        self.x_train = scaler.fit_transform(x_train)
        self.x_test = scaler.fit_transform(x_test)
        self.model = self.model()

    def model(self):
        estimator_KNN = KNeighborsClassifier()
        parameters_KNN = {
            'n_neighbors': range(1, 20),
            'weights': ('uniform', 'distance'),
            'metric': ('minkowski', 'chebyshev', 'euclidean',),
            'p': [1, 2],
            'algorithm': ('auto', 'kd_tree', 'ball_tree')
        }

        kfold = KFold(n_splits=5, shuffle=True, random_state=7)
        grid_knn = GridSearchCV(
            estimator=estimator_KNN,
            param_grid=parameters_KNN,
            scoring='accuracy',
            cv=kfold
        )
        grid_knn.fit(self.x_train, self.y_train)
        model = KNeighborsClassifier().set_params(**grid_knn.best_params_)
        model.fit(self.x_train, self.y_train)
        return model

    def mdi(self):
        rf_model = RandomForestClassifier()
        rf_param = {
            'n_estimators': [500, 800],
            'max_depth': [5, 6, 8, 10],
            'min_samples_split': [10, 20, 30],
        }
        rf_grid = GridSearchCV(rf_model, param_grid=rf_param, scoring='accuracy')
        rf_grid.fit(self.x_train, self.y_train)

        model = RandomForestClassifier().set_params(**rf_grid.best_params_)
        model.fit(self.x_train, self.y_train)
        importance_mdi = pd.DataFrame({'feature': self.X.columns,
                                       'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)

        return importance_mdi

    def pi(self):
        perm = PermutationImportance(self.model, scoring='f1', random_state=0).fit(self.x_test, self.y_test)

        weight = eli5.explain_weights_df(perm, feature_names=self.X.columns.tolist())
        weight_col = weight.feature
        importance_pi = pd.DataFrame({'feature': weight_col,
                                      'importance': weight.weight[:10]}).sort_values(by='importance', ascending=False)
        return importance_pi

    def shap(self):
        shap.initjs()
        x_test_ = pd.DataFrame(self.x_test, columns=self.X.columns)

        explainer = shap.KernelExplainer(self.model.predict, x_test_)
        shap_values = explainer.shap_values(x_test_)

        importance = {}
        for i in range(len(self.X.columns)):
            importance[self.X.columns[i]] = abs(shap_values[:, i]).mean()

        importance_shap = pd.DataFrame([importance]).iloc[0].sort_values(ascending=False).iloc[:10]
        return importance_shap
