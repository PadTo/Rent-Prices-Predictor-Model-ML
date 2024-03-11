# Vilnius Flats Rental Price Prediction

## Overview
This project aims to predict rental prices for flats in Vilnius, Lithuania. Utilizing a dataset from Kaggle, I conducted extensive data cleaning, feature engineering, and machine learning modeling to estimate rental prices accurately.

## Data Source
The initial dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/martynasvarnas/vilnius-flats-for-rent). It consists of 1810 samples, which, after cleaning and filtering, were reduced to 1378 samples.

## Tools and Technologies Used
- **Data Cleaning and Visualization:** Python, Pandas, Matplotlib
- **APIs:**
  - **LocationIQ API:** For fetching latitude and longitude of apartments.
  - **Mapbox API:** To overlay the map over the apartment locations.
- **Machine Learning Libraries:** Scikit-learn, NumPy

## Process

### Data Preprocessing
1. **Fetching Latitude and Longitude:** Utilized LocationIQ API to enrich the dataset with geographical coordinates.
2. **Filtering by City Boundary:** Applied a bounding box to filter out apartments outside Vilnius city limits.
3. **Visualization:** Plotted the apartments on a Mapbox image to visually inspect the distribution.

### Vilnius City Bounding Box Coordinates (Longitude & Latitude)
- North-East Corner: Latitude ~54.75, Longitude ~25.37
- South-West Corner: Latitude ~54.63, Longitude ~25.20

![Alt text](https://github.com/PadTo/Rent-Prices-Predictor-Model-ML/blob/main/Photos_for_git/Display_.png)

### Machine Learning
1. **Model Selection:** Experimented with Linear Regression, SVR (Support Vector Regression), and Random Forest models.
2. **Feature Engineering and Preprocessing:**
    - Implemented one-hot encoding for district features.
    - Applied feature standardization and scaling to improve model performance and ensure that numerical features contribute equally to the model's predictive capability.
3. **Evaluation Metrics:** Used Mean Squared Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE) for model evaluation.

## Key Findings
- **Linear Regression** showed limitations with an average MSE of 62404.46, MAE of 146.15, and MAPE of 0.2147.
- **SVR**, after hyperparameter tuning, indicated poor performance with a test MSE of 60962.34.
- **Random Forest** emerged as the best model, especially when using encoded district features and standardized features. The best Random Forest configuration yielded a final MSE of 11174.27, MAE of 60.67, and MAPE of 0.0905 on the entire dataset.

## Future Directions
Further research could explore the incorporation of additional features, such as proximity to city centres or public transportation, to potentially improve model accuracy

## Model Usage Requirements
To utilize the predictive model effectively, ensure your input data includes only the following columns:
- `price`: The rental price of the flat.
- `rooms`: Number of rooms in the flat.
- `area`: The total area of the flat in square meters.
- `floor`: The floor on which the flat is located.
- `District`: The district within Vilnius where the flat is located.
- `longitude`: The geographical longitude of the flat.
- `latitude`: The geographical latitude of the flat.
- `building size (floors)`: The total number of floors in the building.

It is crucial that the data is correctly formatted and cleaned, adhering to the structure above for optimal model performance.


## Conclusion
The project demonstrated the effectiveness of machine learning in predicting rental prices. The Random Forest model, with careful feature engineering and evaluation, outperformed other models. 

- The final mode evaluation can be found on __main__.py file!
