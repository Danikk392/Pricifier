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
│  • model_exploration.ipynb   │    ← Exploring different Models (GLM, RandomForest, GradientBoosting stacking
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
| **Frontend**        | React.js, Tailwind CSS |
| **Backend**         | Python Flask, SQLAlchemy |
| **Database**        | PostgreSQL/MySQL |
| **AI & NLP**        | Google Cloud Natural Language API, DeepSeek V3 |
| **Web Scraping**    | Selenium WebDriver, BeautifulSoup |
| **Authentication** | Flask-JWT-Extended |
| **API Integration** | OpenRouter API |
| **Deployment**      | TBD (AWS/Google Cloud Platform) |


## **📅 Development Roadmap**

- **Phase 1:** Research & Market Analysis (✅ **Completed**)
- **Phase 2:** Core Backend Development & API Integration (✅ **Completed**)
- **Phase 3:** AI/ML Model Development & Training (✅ **Completed**)
- **Phase 4:** Frontend Dashboard Development (🔄 **In Progress**)
- **Phase 5:** Beta Testing & User Feedback (🔜 **Upcoming**)
- **Phase 6:** Production Deployment & Launch (🔜 **Upcoming**)

## **📊 Key Metrics & Performance**
- **Automated review scraping** from Google Maps using Selenium
- **Real-time sentiment analysis** with Google Cloud NLP
- **AI-powered insights** generation using DeepSeek LLM
- **Comprehensive dashboard** with 8+ analytics endpoints

## **🎯 Target Market**
- **Primary:** Independent restaurant owners (1-10 locations)
- **Secondary:** Restaurant chains and hospitality businesses
- **Tertiary:** Food service management companies
