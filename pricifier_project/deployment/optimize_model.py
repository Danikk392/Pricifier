import pandas as pd
from pricifier.model import ClusterFit, ModelPerCluster, cluster_features, model_types
from pricifier.preprocessor import DataPreprocessor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib

df = pd.read_csv("deployment/Airbnb_Data.csv").sample(500, random_state=42)
X = df.drop(columns=["log_price"])
y = df["log_price"]

preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(X)

print(X_processed.columns.tolist())
clusterer = ClusterFit(cluster_features=cluster_features, n_clusters=3)
cluster_labels = clusterer.fit_predict(X_processed)


selector = ModelPerCluster(
    features=X_processed.columns.tolist(),
    model_types=model_types,
    cluster_labels=cluster_labels,
    n_trials=25,
    timeout=300
)
selector.fit_cluster_models(X_processed, y, cluster_labels)
selector.report()

joblib.dump(selector, "deployment/pricifier/model.pkl")
joblib.dump(preprocessor, "deployment/pricifier/preprocessor.pkl")
joblib.dump(clusterer, "deployment/pricifier/clusterer.pkl")
