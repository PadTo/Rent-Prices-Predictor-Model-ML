# Vilnius Flats Rental Price Prediction

## Overview
This project aims to predict rental prices for flats in Vilnius, Lithuania. Utilizing a dataset from Kaggle, we conducted extensive data cleaning, feature engineering, and machine learning modeling to estimate rental prices accurately.

## Data Source
The initial dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/martynasvarnas/vilnius-flats-for-rent). It consists of 1810 samples, which, after cleaning and filtering, were reduced to 1378 samples.

## Tools and Technologies Used
- **Data Cleaning and Visualization:** Python, Pandas, Matplotlib
- **APIs:**
  - **LocationIQ API:** For fetching latitude and longitude of apartments.
  - **Mapbox API:** To overlay the apartment locations on a map.
- **Machine Learning Libraries:** Scikit-learn, NumPy

## Process

### Data Preprocessing
1. **Fetching Latitude and Longitude:** Utilized LocationIQ API to enrich the dataset with geographical coordinates.
2. **Filtering by City Boundary:** Applied a bounding box to filter out apartments outside Vilnius city limits.
3. **Visualization:** Plotted the apartments on a Mapbox image to visually inspect the distribution.

### Machine Learning
1. **Model Selection:** Experimented with Linear Regression, SVR (Support Vector Regression), and Random Forest models.
2. **Feature Engineering:** Implemented one-hot encoding for district features and standardized numerical features.
3. **Evaluation Metrics:** Used Mean Squared Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE) for model evaluation.

## Key Findings
- **Linear Regression** showed limitations with an average MSE of 62404.46, MAE of 146.15, and MAPE of 0.2147.
- **SVR**, after hyperparameter tuning, indicated a promising direction with a test MSE of 60962.34.
- **Random Forest** emerged as the best model, especially when using encoded district features and standardized features. The best Random Forest configuration yielded a final MSE of 11174.27, MAE of 60.67, and MAPE of 0.0905 on the entire dataset.

## Conclusion
The project demonstrated the effectiveness of machine learning in predicting rental prices. The Random Forest model, with careful feature engineering and evaluation, outperformed other models. Future work could explore more advanced models, additional feature engineering techniques, and larger datasets to improve prediction accuracy.

## How to Run
Instructions on setting up the environment, installing dependencies, and running scripts are provided in separate `INSTALL.md` and `RUNNING.md` files.
