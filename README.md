# **Pricifier**             
*A Full-Stack Machine Learning application for Short Term Rental Price estimation*  

<div align="center" style="background-color: white;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_Bélo.svg/2560px-Airbnb_Logo_Bélo.svg.png" alt="Airbnb Logo" width="300"/>
</div>



## **📌 Overview**
Pricifier is an end-to-end project designed to estimate short-term rental prices using a rich dataset of Airbnb listings. By leveraging **custom model stacking and natural language processing**, Pricifier powers an interactive web application that helps new Airbnb hosts determine competitive pricing for their listings. Predictions are based on key features such as the number of bedrooms, amenities, description sentiment, and geotouristic factors.

## Features

### Core Analytics
- **Custom Model Stacking Pipeline:** Combines multiple models for improved price prediction accuracy  
- **Geotourism-Based Price Modeling:** Incorporates city and neighborhood-level demand factors into pricing logic  
- **NLP-Powered Description Scoring:** Uses sentiment and keyword analysis to extract value signals from listing descriptions  
- **Amenity Parsing and Encoding:** Identifies and encodes listing amenities using TF-IDF and clustering for richer feature sets

### Web Interface
- **Interactive Pricing Interface:** Web app where users input listing details to receive a predicted nightly price
- **Cluster-Based Feature Segmentation:** Listings are grouped into meaningful segments to refine model predictions

### Business Intelligence
- **Data-Driven Pricing Suggestions:** Helps new Airbnb hosts understand optimal pricing ranges based on similar listings
- **Market Positioning Insights:** Suggests how different features (e.g. sentiment, amenities) impact price competitiveness  
- **Outlier Detection Capability:** Handles luxury or undersupplied listings through cluster-aware modeling logic  
 


## **🏗 System Architecture**
```

┌──────────────────────────────┐
│   🧪 Jupyter Modeling Layer  │    ← Heavy modeling & experimentation
│  (machine_learning/,         │
│   processing/, etc.)         │
│  ─────────────────────────   │
│  • data_processing.ipynb     │    ← Encoding/imputing data pipeline   
│  • linear_processing.ipynb   │    ← GLM modeling  
│  • model_exploration.ipynb   │    ← Exploring different Models (GLM, RandomForest, GradientBoosting)
│  • cluster_fit.ipynb         │    ← KMeans clustering segmentation and model stacking  
│                              │
│  ➤ Output:                   │
│  - model.pkl                 │
│  - clusterer.pkl             │
│  - preprocessor.pkl          │
└────────────▲─────────────────┘
             │
             │ Saved using:
             ▼
┌──────────────────────────────┐
│  🔧 Custom Model Scripts     │
│  (deployment/pricifier/)     │
│  • save_model.py             │ ← Wraps pipeline & saves model  
│  • save_preprocessor.py      │ ← Preprocess pipeline export  
└────────────▲─────────────────┘
             │
             │ Used by:
             ▼
┌──────────────────────────────┐
│   🌐 Django Web Application  │
│  (deployment/pricifier/)     │
│  ─────────────────────────   │
│  • views.py – runs prediction│
│  • forms.py – user input     │
│  • utils.py – model helpers  │
│  • templates/ – frontend UI  │
│     - index.html             │
│     - predict.html           │
│  • static/style.css          │
└────────────▲─────────────────┘
             │
             │ Served locally using:
             ▼
┌──────────────────────────────┐
│       ⚙ Runserver (Dev)      │
│  python manage.py runserver  │
│  SQLite3 as local database   │
└──────────────────────────────┘

```


## **📐 Tech Stack**

| **Layer**           | **Technology** |
|---------------------|----------------|
| **Frontend**        | HTML5, CSS, Django Templates |
| **Backend**         | Python, Django Framework |
| **Modeling & ML**        | Scikit-learn, Pandas, NumPy, Optuna |
| **NLP**        | Custom sentiment scoring, TF-IDF, TexBlob, VADER |
| **Clustering & Stacking**    | KMeans, Random Forest, Ridge, Lasso, StackingRegressor |
| **Data Exploration** | Jupyter Notebooks, Matplotlib |
| **Model Persistence** | joblib for .pkl exports |
| **Deployment (Local)**      | Django runserver, SQLite3 |
| **Version Control**      | Git, GitHub |


## **📅 Development Roadmap**

- **Phase 1:** Exploratory Data Analysis & Feature Engineering (✅ **Completed**)
- **Phase 2:** Model Training, Hyperparameter Tuning & Clustering(✅ **Completed**)
- **Phase 3:** Internal Testing & Model Evaluation (✅ **Completed**)
- **Phase 4:** Preprocessing Pipeline & Model Serialization (.pkl exports) (✅ **Completed**)
- **Phase 5:** Django Backend Integration & Prediction Logic (✅ **Completed**)
- **Phase 6:** Frontend Web Interface (HTML/CSS + Templates) (✅ **Completed**)
- **Phase 8:** Cloud Deployment (e.g. Render, Heroku, or AWS) (🔜 **Potentially**)

## **📊 Key Metrics & Performance**
- **Improved prediction error** from **70% to 22%** using stacked models and cluster-aware feature selection
- **Model stacking** using Ridge, Lasso, and Random Forest regressors for enhanced accuracy
- Clustering (KMeans) used to segment listings by feature profiles for better generalization
- Sentiment scoring & **TF-IDF** on listing descriptions to capture narrative-driven price signals
- Processed **74,000+ listings** across major U.S. cities for training and validation
  
