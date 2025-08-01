import joblib
import pandas as pd
from pricifier.preprocessor import DataPreprocessor

# Load data
df = pd.read_csv("deployment/Airbnb_Data.csv").sample(500, random_state=42)
X_train = df.drop(['log_price'], axis=1)

# Fit preprocessor
preprocessor = DataPreprocessor()
preprocessor.fit(X_train)

# Save correctly using real import path
joblib.dump(preprocessor, 'deployment/pricifier/preprocessor.pkl')
