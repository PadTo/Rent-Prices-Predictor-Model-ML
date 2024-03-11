import joblib
import pandas as pd
import numpy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load the saved model
loaded_model = joblib.load("model/random_forest_model.pkl")


def preprocessing(data):
    # One-hot encode the categorical feature 'District'
    enc = OneHotEncoder()
    encoded_data = enc.fit_transform(data[["District"]])

    # Convert the encoded data into a DataFrame
    df_encoded_data = pd.DataFrame(
        encoded_data.toarray(), columns=enc.get_feature_names_out())

    # Concatenate the original data with the encoded features
    data_processed = pd.concat(
        [data.drop(["District"], axis=1), df_encoded_data], axis=1)

    return data_processed


# Load your new data into a pandas DataFrame
new_data = pd.read_csv(
    "datasets/Vilnius_rent_prices_filtered_ML(With_Districts).csv")

# Preprocess the new data using the preprocessing function
preprocessed_data = preprocessing(new_data)

# Separate the features from the target variable 'price'
X_new = preprocessed_data.drop("price", axis=1)
# Corrected to use the price as the target variable
y_true = preprocessed_data["price"]

# Use the loaded model to generate predictions for the preprocessed new data
predictions = loaded_model.predict(X_new)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_true, predictions)
print("Mean Squared Error (MSE):", mse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true, predictions)
print("Mean Absolute Error (MAE):", mae)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_true, predictions)
print("Mean Absolute Percentage Error (MAPE):", mape)
