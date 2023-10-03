from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import shap
import pandas as pd

class LGBMWithSHAP:
    
    def __init__(self, df, target_col, cat_features, param_grid):
        self.df = df
        self.target_col = target_col
        self.cat_features = cat_features
        self.param_grid = param_grid
        self.model = None
        self.shap_values = None
    
    def prepare_data(self):
        for col in self.cat_features:
            self.df[col] = self.df[col].astype('category')
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        pipeline = Pipeline([('lgbm', LGBMRegressor())])
        grid_search = GridSearchCV(pipeline, self.param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
    
    def calculate_shap_values(self):
        if self.model is None:
            print("Model not trained yet!")
            return
        explainer = shap.TreeExplainer(self.model.named_steps['lgbm'])
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.shap_values = explainer.shap_values(X_test)
    
    def summary_plot(self):
        if self.shap_values is None:
            print("SHAP values not calculated yet!")
            return
        X_train, X_test, y_train, y_test = self.prepare_data()
        shap.summary_plot(self.shap_values, X_test, plot_type="bar")
