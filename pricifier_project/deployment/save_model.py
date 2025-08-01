import joblib

model = joblib.load("deployment/pricifier/model.pkl")
preprocessor = joblib.load("deployment/pricifier/preprocessor.pkl")
clusterer = joblib.load("deployment/pricifier/clusterer.pkl")

# You can now use these inside your Django view or test prediction
