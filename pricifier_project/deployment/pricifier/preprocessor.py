from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def comma_tokenizer(x):
    return x.split(',')

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, impute_strategy="mean", encode_type="onehot"):
        self.impute_strategy = impute_strategy
        self.encode_type = encode_type
        self.analyzer = SentimentIntensityAnalyzer()
        self.tfidf_top_features = None
        self.fitted = False 
        self.scaler = MinMaxScaler()
        self.tfidf_scaler = StandardScaler()
        self.amenity_scaler = StandardScaler()
        self.tfidf_matrix = None
        self.vectorizer = TfidfVectorizer(tokenizer=comma_tokenizer, lowercase=False)
        self.final_feature_names = []
        self.cluster_features = [
            'room_type',
            'distance_to_city_center',
            'sentiment',
            'objectivity',
             'beds',
            'n_amenities',
            'amenity_score_normalized', 
            'review_scores_rating', 
            'city_value_score', 'city_expense_score'
            ]

        self.imputers = {
            "host_response_rate": SimpleImputer(strategy="constant", fill_value=0),
            "review_scores_rating": SimpleImputer(strategy="median"),
            "first_review": SimpleImputer(strategy="constant", fill_value="2000-01-01"),
            "last_review": SimpleImputer(strategy="constant", fill_value="2000-01-01"),
            "thumbnail_url": SimpleImputer(strategy="constant", fill_value="missing_thumbnail"),
            "neighbourhood": SimpleImputer(strategy="constant", fill_value="missing"),
            "zipcode": SimpleImputer(strategy="constant", fill_value="00000"),
            "bathrooms": SimpleImputer(strategy="median"),
            "bedrooms": SimpleImputer(strategy="median"),
            "beds": SimpleImputer(strategy="median"),
            "host_has_profile_pic": SimpleImputer(strategy="constant", fill_value="f"),
            "host_identity_verified": SimpleImputer(strategy="constant", fill_value="f"),
            "host_since": SimpleImputer(strategy="constant", fill_value="2000-01-01"),
        }

        self.encoders = {
            "mapping": {
                "room_type": {
                    "Private room": 0,
                    "Entire home/apt": 1,
                    "Shared room": 2
                },
                "bed_type": {
                    "Real Bed": 5,
                    "Futon": 4,
                    "Pull-out Sofa": 3,
                    "Airbed": 2,
                    "Couch": 1,
                },
                "cancellation_policy_map_s": {
                    'flexible': 0.0,
                    'moderate': 0.33,
                    'strict': 0.66,
                    'super_strict_30': 0.83,
                    'super_strict_60': 1.0
                },
                "cancellation_policy_map_f": {
                    'flexible': 1.0,
                    'moderate': 0.66,
                    'strict': 0.33,
                    'super_strict_30': 0.16,
                    'super_strict_60': 0.0
                }
            }
        }

        self.bool_cols = ["host_has_profile_pic", "host_identity_verified", "instant_bookable"]
        self.map_cols = ["bed_type", "room_type"]

        self.city_sentiment = {
            "NYC": 0.75, "LA": 0.78, "SF": 0.85, "DC": 0.73, "Chicago": 0.60, "Boston": 0.90
        }

        self.city_expense_worth = {
            "NYC": 0.55, "LA": 0.58, "SF": 0.50, "DC": 0.78, "Chicago": 0.72, "Boston": 0.75
        }

        self.city_centers = {
            'NYC': (40.7549, -73.984),
            'LA': (34.0557, -118.2488),
            'SF': (37.7876, -122.4066),
            'DC': (38.9037, -77.0363),
            'Chicago': (41.8757, -87.6243),
            'Boston': (42.3555, -71.0565)
        }

    def sentiment_score(self, X):
        def compute_sentiment(text):
            if not isinstance(text, str) or text.strip() == "":
                return 0
            return self.analyzer.polarity_scores(text)['compound']
        X['sentiment'] = X['description'].apply(compute_sentiment)
        return X

    def objectivity_score(self, X):
        def compute_obj(text):
            if not isinstance(text, str): return 0
            return 1 - TextBlob(text).sentiment.subjectivity
        X['objectivity'] = X['description'].apply(compute_obj)
        return X

    def combine_sentiment_subjectivity(self, sentiment, objectivity, sentiment_weight=0.7, objectivity_weight=0.3):
        sentiment_norm = (sentiment + 1) / 2
        return sentiment_weight * sentiment_norm + objectivity_weight * objectivity

    def day_since(self, start, end):
        start = pd.to_datetime(start, errors="coerce")
        end = pd.to_datetime(end, errors="coerce")
        return (end - start).days if pd.notnull(start) and pd.notnull(end) else -1

    def _compute_amenity_score(self, X, top_k=30):
        pet_map = [
            "Pets live on this property", "Pets allowed", "Dog(s)", "Cat(s)", "Other pet(s)"
        ]
        amenities_map = {
            "Wireless Internet": "Internet",
            "Dryer": "Dryer/Washer",
            "Washer": "Dryer/Washer",
            "Dishwasher": "Dryer/Washer",
            "Central Heating": "Heating",
            **{p: "Pet-Friendly" for p in pet_map}
        }

        def map_amenities(amenity_list):
            return [amenities_map.get(a.strip().strip('"'), a.strip().strip('"')) for a in amenity_list]

        X = X.copy()

        # Ensure amenities are in string format
        X['amenities'] = X['amenities'].fillna('').astype(str)

        # Create 'split_amenities' if it doesn't exist
        if 'split_amenities' not in X.columns:
            X['split_amenities'] = X['amenities'].str.replace(r'[\{\}"\']', '', regex=True).str.split(',')


        if 'amenities_str' not in X.columns:
            X['standard_amenities'] = X['split_amenities'].apply(map_amenities)
            X['amenities_str'] = X['standard_amenities'].apply(lambda x: ','.join(x))

        if not self.fitted:
            assert 'amenities_str' in X.columns, "Missing 'amenities_str' column in input data"
            tfidf = self.vectorizer.fit_transform(X['amenities_str'])
            tfidf_df = pd.DataFrame(tfidf.toarray(), columns=self.vectorizer.get_feature_names_out(), index=X.index)

            # Store top features only ONCE
            top_amenities = tfidf_df.mean().sort_values(ascending=False).head(top_k).index.tolist()
            self.tfidf_top_features = top_amenities
            tfidf_top_df = tfidf_df[top_amenities]
            self.scaler.fit(tfidf_top_df)
            self.fitted = True
        else:
            tfidf = self.vectorizer.transform(X['amenities_str'])
            tfidf_df = pd.DataFrame(
                tfidf.toarray(),
                columns=self.vectorizer.get_feature_names_out(),
                index=X.index
            )
            tfidf_top_df = tfidf_df[self.tfidf_top_features]

            self.tfidf_matrix = tfidf_top_df

        X['amenity_score'] = tfidf_top_df.sum(axis=1)
        X['amenity_score_normalized'] = self.amenity_scaler.fit_transform(X[['amenity_score']])

        return pd.concat([X, tfidf_top_df], axis=1)

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def add_distance_to_city_center(self, X, city_col='city', lat_col='latitude', lon_col='longitude'):
        def compute_distance(row):
            city = row[city_col]
            if city in self.city_centers:
                center_lat, center_lon = self.city_centers[city]
                return self.haversine(row[lat_col], row[lon_col], center_lat, center_lon)
            else:
                return np.nan
        X = X.copy()
        X['distance_to_city_center'] = X.apply(compute_distance, axis=1)
        return X

    def fit(self, X, y=None):
        X = X.copy()
        for col, imputer in self.imputers.items():
            imputer.fit(X[[col]])
            X[col] = imputer.transform(X[[col]]).ravel()

        X = self._compute_amenity_score(X)
        X = self.add_distance_to_city_center(X)
        X = self.objectivity_score(X)
        X = self.sentiment_score(X)
        X['n_amenities'] = X['amenities'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        # X['amenity_score_normalized'] = self.amenity_scaler.transform(X[['amenity_score']])
        X['city_value_score'] = X['city'].map(self.city_sentiment)
        X['city_expense_score'] = X['city'].map(self.city_expense_worth)        
        
        self.bool_encoders = {}
        for col in self.bool_cols:
            enc = OrdinalEncoder(categories=[["f", "t"]], dtype=int, handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(X[[col]])
            self.bool_encoders[col] = enc

        # Preserve all numeric + cluster features
        cluster_cols = [col for col in self.cluster_features if col in X.columns]
        other_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        self.final_feature_names = list(set(cluster_cols + other_cols))

        self.fitted = True
        return self

    def transform(self, X):
        X = X.copy()

        for col, imputer in self.imputers.items():
            if col not in X.columns:
                X[col] = imputer.transform([[None]])[0][0]
            else:
                X[col] = imputer.transform(X[[col]]).ravel()

        X['first_review'] = pd.to_datetime(X['first_review'], errors="coerce")
        X['last_review'] = pd.to_datetime(X['last_review'], errors="coerce")
        X['missing_review_dates'] = X['first_review'].isna() | X['last_review'].isna()

        mask = X['first_review'].notna() & X['last_review'].notna()
        X['review_gap_days'] = np.nan
        X.loc[mask, 'review_gap_days'] = (X.loc[mask, 'last_review'] - X.loc[mask, 'first_review']).dt.days
        X['review_gap_days'] = X['review_gap_days'].fillna(-1)

        X['host_response_rate'] = (
            X['host_response_rate']
            .replace("None", "0")  
            .fillna("0")           
            .astype(str)
            .str.rstrip('%')
            .astype(float)
            )

        if 'cleaning_fee' in X.columns:
            X['cleaning_fee'] = X['cleaning_fee'].fillna(0).astype(int)
        else:
            X['cleaning_fee'] = 0

        for col in self.bool_cols:
            if col not in X.columns:
                X[col] = "f"
            X[col] = self.bool_encoders[col].transform(X[[col]]).ravel()

        for col, mapping in self.encoders["mapping"].items():
            if col in X.columns:
                X[col] = X[col].map(mapping)

        if 'latitude' in X.columns:
            X['lat_bin'] = pd.cut(X['latitude'], bins=10)
        if 'longitude' in X.columns:
            X['long_bin'] = pd.cut(X['longitude'], bins=10)

        if 'amenities' in X.columns:
            X['split_amenities'] = X['amenities'].fillna('').apply(lambda x: str(x).strip("{}").split(','))
            X['n_amenities'] = X['amenities'].fillna('').apply(lambda x: len(str(x).strip("{}").split(',')))

        X = self.sentiment_score(X)
        X = self.objectivity_score(X)
        X['description_score'] = X.apply(
            lambda row: self.combine_sentiment_subjectivity(row['sentiment'], row['objectivity']), axis=1)

        if 'cancellation_policy' in X.columns:
            X['luxury_policy_flag'] = X['cancellation_policy'].isin(['super_strict_30', 'super_strict_60']).astype(int)
        else:
            X['luxury_policy_flag'] = 0

        X['city_value_score'] = X['city'].map(self.city_sentiment)
        X['city_expense_score'] = X['city'].map(self.city_expense_worth)

        X['days_between_reviews'] = X.apply(
            lambda row: self.day_since(row['first_review'], row['last_review']), axis=1)
        
        most_recent_possible = pd.to_datetime("2017-10-04")

        X['host_tenure'] = X.apply(
            lambda row: self.day_since(row['host_since'], row['last_review'])
            if pd.notnull(row['last_review']) and 
                pd.to_datetime(row['last_review']) >= pd.to_datetime(row['host_since'])
            else self.day_since(row['host_since'], most_recent_possible),
            axis=1
        )
        X = self.add_distance_to_city_center(X)

        if 'split_amenities' in X.columns:
            X = self._compute_amenity_score(X)
        
        X['amenity_score'] = self.tfidf_matrix.sum(axis=1)
        X['amenity_score_normalized'] = self.amenity_scaler.transform(X[['amenity_score']])

        for col in self.final_feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.final_feature_names]

        drop_cols = ['id', 'name', 'thumbnail_url', 'neighbourhood',
                     'first_review', 'host_since', 'last_review', 'zipcode', 'missing_review_dates']
        X = X.drop(columns=drop_cols, errors='ignore')

        # Final return: numeric-only DataFrame for safety in pipelines
        return X.select_dtypes(include=[np.number])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)