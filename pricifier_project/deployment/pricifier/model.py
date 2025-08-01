from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import pandas as pd
import optuna 
from sklearn.ensemble import StackingRegressor

cluster_features = [
'room_type',
'distance_to_city_center',
 'sentiment',
 'objectivity',
 'beds',
 'n_amenities',
 'amenity_score_normalized', 
 'review_scores_rating', 
 'city_value_score', 'city_expense_score']

class ClusterFit:
    def __init__(self, cluster_features, n_clusters=3):
        self.cluster_features = cluster_features
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.fitted = False

    def fit(self, X):
        X_cluster = self.scaler.fit_transform(X[self.cluster_features])
        self.kmeans.fit(X_cluster)
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise ValueError("ClusterHelper must be fitted before calling predict()")
        X_cluster = self.scaler.transform(X[self.cluster_features])
        return self.kmeans.predict(X_cluster)

    def fit_predict(self, X):
        self.fit(X)
        return self.kmeans.predict(self.scaler.transform(X[self.cluster_features]))
    

class ModelPerCluster:
    def __init__(self, features, model_types, cluster_labels, n_trials=30, timeout=300):
        self.features = features
        self.n_clusters = 3
        self.model_types = model_types
        self.n_trials = n_trials
        self.timeout = timeout
        self.cluster_labels = cluster_labels

        self.scalar = StandardScaler()
        self.cluster_models = {}
        self.cluster_studies = {}
        self.cluster_rmses = {}
        self.cluster_model_types = {}

    def fit_cluster_models(self, X, y, cluster_labels):
        for cluster_id in range(self.n_clusters):
            print(f"\n>>> Tuning cluster {cluster_id}")

            mask = cluster_labels == cluster_id
            X_cluster = X[mask].copy()
            y_cluster = y[mask].copy()

            best_rmse = float('inf')
            best_model = None
            best_study = None
            best_model_type = None

            optuna_params = {}

            for name, (model_class, param_space) in self.model_types.items():
                if name == "Stacking":
                    continue

                def theobjective(trial):
                    params = param_space(trial)
                    X_train, X_val, y_train, y_val = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)
                    model = model_class(**params)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_val, preds))
                    return rmse
                
                study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(theobjective, n_trials=self.n_trials, timeout=self.timeout)

                if study.best_value < best_rmse:
                    best_rmse = study.best_value
                    best_study = study
                    best_model_type = name
                    best_model = model_class(**study.best_params)
                    best_model.fit(X_cluster, y_cluster)
                
                optuna_params[name] = study.best_params

            if cluster_id in [0, 1, 2]:
                base_models = []
                for base_name in ["RandomForest", "GradientBoosting", "Ridge"]:
                    if base_name in self.model_types:
                        model_class, param_space = self.model_types[base_name]
                        params = optuna_params.get(base_name, {})
                        base_model = model_class(**params)
                        base_model.fit(X_cluster, y_cluster)
                        base_models.append((base_name, base_model))

                stacked_model = StackingRegressor(estimators=base_models, final_estimator=Ridge())
                stacked_model.fit(X_cluster, y_cluster)
                stacked_preds = stacked_model.predict(X_cluster)
                stacked_rmse = np.sqrt(mean_squared_error(y_cluster, stacked_preds))

                if stacked_rmse < best_rmse:
                    best_rmse = stacked_rmse
                    best_model_type = "Stacking"
                    best_model = stacked_model
                    best_study = None

            self.cluster_models[cluster_id] = best_model
            self.cluster_studies[cluster_id] = best_study
            self.cluster_rmses[cluster_id] = best_rmse
            self.cluster_model_types[cluster_id] = best_model_type
            
    def predicts(self, X, cluster_labels):
        preds = np.zeros(len(X))
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            if np.sum(mask) == 0:
                continue
            X_cluster = X[mask]
            model = self.cluster_models[cluster_id]
            preds[mask] = model.predict(X_cluster)
        return preds
        
    def report(self):
        print("\n--- Cluster Summary ---")
        for cluster_id in range(self.n_clusters):
            print(f"Cluster {cluster_id}: {self.cluster_model_types[cluster_id]} | RMSE: {self.cluster_rmses[cluster_id]:.4f}")

def rf_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }

def gbr_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
    }

def ridge_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.01, 10.0),
    }

model_types = {
    "RandomForest": (RandomForestRegressor, rf_space),
    "Ridge": (Ridge, ridge_space),
    "GradientBoosting": (GradientBoostingRegressor, gbr_space)
}
