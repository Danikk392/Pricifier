# **Pricifier**             
*A Full-Stack Machine Learning application for Short Term Rental Price estimation*  

<div style="width: 100; display: flex; justify-content: flex-end; background-color: white;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_Bélo.svg/2560px-Airbnb_Logo_Bélo.svg.png"
       alt="Airbnb Logo"
       width="300" />
</div>



## **📌 Overview**
Pricifier is an end-to-end project designed to estimate short-term rental prices using a rich dataset of Airbnb listings. By leveraging **custom model stacking and natural language processing**, Pricifier powers an interactive web application that helps new Airbnb hosts determine competitive pricing for their listings. Predictions are based on key features such as the number of bedrooms, amenities, description sentiment, and geotouristic factors.

## **🎯 Problem Statement**
Restaurant owners highly value customer feedback, especially from Google Reviews, but this feedback is often:
- Scattered across multiple platforms
- Difficult to analyze at scale
- Time-consuming to extract actionable insights
- Hard to track trends and performance metrics

## **💡 Solution**
TableTalk solves these challenges by providing:
- **Automated review aggregation** from Google Maps and other platforms
- **AI-powered sentiment analysis** and entity extraction using Google Cloud NLP
- **Real-time performance dashboards** with comprehensive analytics
- **Actionable AI insights** generated using advanced language models

## Features

### Core Analytics
- **Smart Review Classification**: Categorize reviews by sentiment and topic  
- **AI-Powered Sentiment Analysis**: Track customer satisfaction with Google Cloud NLP  
- **Multi-Platform Aggregation**: Scrape Google Maps using Selenium  
- **Entity Extraction**: Identify key topics, menu items, and business aspects  

### Dashboard & Visualization
- **Analytics Dashboard**: Centralized view of all reviews  
- **Performance Metrics Tracking**: Monitor rating trends, review volume, and sentiment over time  
- **Review Segmentation**: Segment reviews into Highly Positive, Critical, and Suggestions  


### Business Intelligence
- **AI-Generated Insights**: Business recommendations powered by DeepSeek LLM  
- **Topic-Based Ratings**: Track performance across specific aspects of the business  
- **Critical Review Detection**: Highlight and prioritize negative feedback  
 


## **🏗 System Architecture**

```
┌─────────────────────┐    JWT Auth     ┌─────────────────────┐    API Calls    ┌─────────────────────┐
│      Frontend       │    + REST API   │     Backend API     │                 │    External APIs    │
│   React.js +        │◄───────────────►│   Flask +           │◄───────────────►│                     │
│   Tailwind CSS      │                 │   SQLAlchemy        │                 │  ┌───────────────┐  │
│                     │                 │                     │                 │  │ Google Cloud  │  │
│ ┌─────────────────┐ │                 │ ┌─────────────────┐ │                 │  │ NLP API       │  │
│ │ Dashboard UI    │ │                 │ │ Auth Service    │ │                 │  │ • Sentiment   │  │
│ │ Analytics       │ │                 │ │ Business Logic  │ │                 │  │ • Entities    │  │
│ │ Review Mgmt     │ │                 │ │ Dashboard APIs  │ │                 │  └───────────────┘  │
│ └─────────────────┘ │                 │ └─────────────────┘ │                 │                     │
└─────────────────────┘                 └─────────────────────┘                 │  ┌───────────────┐  │
                                                   │                             │  │ OpenRouter    │  │
                                                   │                             │  │ (DeepSeek V3) │  │
                                                   │                             │  │ • AI Insights │  │
                                                   │                             │  │ • Summary     │  │
                                                   ▼                             │  └───────────────┘  │
                                        ┌─────────────────────┐                 └─────────────────────┘
                                        │   Data Processing   │                            │
                                        │     Pipeline        │                            │
                                        └─────────────────────┘                            │
                                                   │                                       │
                                                   ▼                                       │
┌─────────────────────┐    Web Scraping ┌─────────────────────┐    Raw Data    ┌─────────────────────┐
│     Data Sources    │◄────────────────│   Scraping Layer   │───────────────►│    Database Layer   │
│                     │                 │                     │                 │                     │
│ ┌─────────────────┐ │                 │ ┌─────────────────┐ │                 │ ┌─────────────────┐ │
│ │ Google Maps     │ │                 │ │ Selenium        │ │                 │ │ PostgreSQL/     │ │
│ │ Review Pages    │ │                 │ │ WebDriver       │ │                 │ │ MySQL           │ │
│ │                 │ │                 │ │ • Rate Limiting │ │                 │ │                 │ │
│ └─────────────────┘ │                 │ │ • Smart Parsing │ │                 │ │ ┌─────────────┐ │ │
│                     │                 │ └─────────────────┘ │                 │ │ │   Tables    │ │ │
│ ┌─────────────────┐ │                 │                     │                 │ │ │ • Users     │ │ │
│ │ Other Review    │ │                 │ ┌─────────────────┐ │                 │ │ │ • Business  │ │ │
│ │ Platforms       │ │                 │ │ BeautifulSoup   │ │                 │ │ │ • Reviews   │ │ │
│ │ (Future)        │ │                 │ │ • HTML Parsing  │ │                 │ │ │ • Insights  │ │ │
│ └─────────────────┘ │                 │ │ • Data Cleaning │ │                 │ │ └─────────────┘ │ │
└─────────────────────┘                 │ └─────────────────┘ │                 │ └─────────────────┘ │
                                        └─────────────────────┘                 └─────────────────────┘
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
