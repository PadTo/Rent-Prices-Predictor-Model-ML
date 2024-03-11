# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import reciprocal
import joblib


# Load the dataset
df = pd.read_csv(
    "datasets/Vilnius_rent_prices_filtered_ML(With_Districts).csv")

# Basic data exploration
print(df.describe())  # Descriptive statistics for numerical features
print(df.head())  # Display the first few rows of the dataframe

# Analyze the distribution of districts
# Count unique districts
print(df["District"].value_counts().sort_index().count())

# Price category creation for stratified sampling
df["price category"] = pd.cut(df["price"],
                              bins=[0, 500, 650, 830, np.inf],
                              labels=[1, 2, 3, 4])

# Visualize the price categories
print(df["price category"].value_counts().plot.bar(rot=0))

# Original distribution of numerical features
df.hist(bins=50, figsize=(14, 7))
plt.show()


# Data transformation - Applying log transformation to 'area' and 'price'
df["area"] = df["area"].apply(lambda area_: np.log(area_))
df["price"] = df["price"].apply(lambda price_: np.log(price_))

# Visualize the distribution after log transformation
df.hist(bins=50, figsize=(14, 7))
plt.show()

# One-hot encoding for categorical feature 'District'
enc = OneHotEncoder()
encoded_data = enc.fit_transform(df[["District"]])
# Print the names of the created one-hot encoded features
print(enc.get_feature_names_out())

# Drop columns that you don't want to scale or transform
df = df.drop(["District", "price category"], axis=1)

# Create a copy of df for safe manipulation
df_copy = df.copy()

# Identify the features to scale, excluding 'price'
features_to_scale = df_copy.columns.drop(['price'])

# Initialize the scaler
scaler = StandardScaler()

# Apply scaling to the selected features
df_copy[features_to_scale] = scaler.fit_transform(df_copy[features_to_scale])

# Assuming 'price' is to be the first column, concatenate 'price' with the scaled features
df_scaled = pd.concat([df[['price']], df_copy.drop(['price'], axis=1)], axis=1)

print(df_scaled.head())
# Visualize the distribution after scaling
df_scaled.hist(bins=50, figsize=(14, 7))
plt.show()

# Convert the encoded data into a pandas DataFrame
encoded_data_df = pd.DataFrame(
    encoded_data.toarray(), columns=enc.get_feature_names_out())
# Display the first few rows of the encoded dataframe
print(encoded_data_df.head())

# Combine the original dataframe (excluding 'District') with the encoded DataFrame
df_combined = pd.concat([df_scaled, encoded_data_df], axis=1)
# Display the first few rows of the combined dataframe
print(df_combined.head())

# Leaving 10% of the data for final evaluation of the model
df_train_test, df_final_val = train_test_split(
    df_combined, test_size=0.1, random_state=42)

model = LinearRegression()
mean_square_results = []
mean_abs_results = []
mean_square_perc_results = []

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kfolds.split(df_train_test):
    X_train, X_test = df_train_test.drop(["price"], axis=1).iloc[train_index], df_train_test.drop([
        "price"], axis=1).iloc[test_index]
    y_train, y_test = df_train_test["price"].iloc[train_index], df_train_test["price"].iloc[test_index]

    # Linear Model #
    linear_model = model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)

    # Invert the log transformation for predictions and actual values
    y_pred = np.exp(y_pred)
    y_test = np.exp(y_test)

    mse = mean_squared_error(
        y_pred=y_pred,
        y_true=y_test)
    mae = mean_absolute_error(
        y_pred=y_pred,
        y_true=y_test)
    mape = mean_absolute_percentage_error(
        y_pred=y_pred,
        y_true=y_test)

    mean_square_results.append(mse)
    mean_abs_results.append(mae)
    mean_square_perc_results.append(mape)

# Calculate average performance metrics over k-folds
average_mse = np.mean(mean_square_results)
average_mae = np.mean(mean_abs_results)
average_mape = np.mean(mean_square_perc_results)

print(f"Average MSE (Linear): {average_mse}")
print(f"Average MAE (Linear): {average_mae}")
print(f"Average MAPE (Linear): {average_mape}")

# Support Vector Machine (Regression) #

# Features matrix X: Dropping the target variable 'price' and non-feature column 'price category'
X_train = df_train_test.drop(["price"], axis=1)
y_train = df_train_test["price"]

# Define parameter distributions for SVR
param_distributions = {
    'C': reciprocal(1e-4, 1e4),
    'gamma': reciprocal(1e-4, 1e1),
    'kernel': ['rbf', 'poly']  # You can add more kernels to try
}

# Initialize SVR model
model_svr = SVR()

# Perform RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=model_svr,
    param_distributions=param_distributions,
    n_iter=8,  # Number of parameter settings to sample
    cv=4,  # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit the randomized search model
random_search.fit(X_train, y_train)

# Display best hyperparameters and score
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Get the best model from the search
best_model = random_search.best_estimator_

# Evaluate the best model on the final validation set
X_test = df_final_val.drop(["price"], axis=1)
y_test = df_final_val["price"]

y_pred = best_model.predict(X_test)

mse = mean_squared_error(np.exp(y_test),
                         np.exp(y_pred))
print("Test MSE:", mse)

# Testing without districts

encoded_district_columns = enc.get_feature_names_out()

df_train_test_wo_districts = df_train_test.drop(
    encoded_district_columns, axis=1)

df_final_val_wo_districts = df_final_val.drop(
    encoded_district_columns, axis=1)

X_train_wo_districts = df_train_test_wo_districts.drop(["price"], axis=1)
y_train_wo_districts = df_train_test_wo_districts["price"]

X_final_val_wo_districts = df_final_val_wo_districts.drop(["price"], axis=1)
y_final_val_wo_districts = df_final_val_wo_districts["price"]

model_wo_districts = LinearRegression()
model_wo_districts.fit(X_train_wo_districts, y_train_wo_districts)

y_pred_wo_districts = model_wo_districts.predict(X_final_val_wo_districts)

y_pred_exp = np.exp(y_pred_wo_districts)
y_test_exp = np.exp(y_final_val_wo_districts)

mse_wo_districts = mean_squared_error(y_test_exp, y_pred_exp)
print("Test MSE (Without Districts):", mse_wo_districts)

# Perform RandomizedSearchCV on the model without districts
random_search.fit(X_train_wo_districts, y_train_wo_districts)

# Display best hyperparameters and score
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Get the best model from the search
best_model = random_search.best_estimator_

# Evaluate the best model on the final validation set without districts
X_test = X_final_val_wo_districts
y_test = y_final_val_wo_districts

y_pred = best_model.predict(X_test)

mse = mean_squared_error(np.exp(y_test),
                         np.exp(y_pred))
print("Test MSE:", mse)

# Initialize the Random Forest regressor with default parameters
rf_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)

# Fit the model to the training data without districts
rf_regressor.fit(X_train_wo_districts, y_train_wo_districts)

# Predict on the final validation set without districts
y_pred_rf = rf_regressor.predict(X_final_val_wo_districts)

# The target variable was log-transformed, thus, apply the inverse transformation
y_pred_rf_exp = np.exp(y_pred_rf)
y_final_val_exp = np.exp(y_final_val_wo_districts)

# Calculate performance metrics
mse_rf = mean_squared_error(y_final_val_exp, y_pred_rf_exp)
print("Random Forest Test MSE:", mse_rf)


# Initialize the Random Forest regressor with default parameters
rf_regressor_districts = RandomForestRegressor(random_state=42, n_jobs=-1)

# Features matrix X: Dropping the target variable 'price'
X_train_districts = df_train_test.drop(["price"], axis=1)
# Target vector y: Using the 'price' column as the target variable
y_train_districts = df_train_test["price"]

# Fit the model to the training data with districts
rf_regressor_districts.fit(X_train_districts, y_train_districts)

# Predict on the final validation set with districts
y_pred_rf_districts = rf_regressor_districts.predict(
    df_final_val.drop(["price"], axis=1))

# If your target variable was log-transformed, apply the inverse transformation
y_pred_rf_districts_exp = np.exp(y_pred_rf_districts)
y_final_val_exp_districts = np.exp(df_final_val["price"])

# Calculate performance metrics
mse_rf_districts = mean_squared_error(
    y_final_val_exp_districts, y_pred_rf_districts_exp)
print("Random Forest Test MSE (with Districts):", mse_rf_districts)


# Initialize the Random Forest regressor with default parameters
rf_regressor_districts = RandomForestRegressor(random_state=42, n_jobs=-1)

# Features matrix X: Dropping the target variable 'price' and latitude and longitude
X_train_districts = df_train_test.drop(
    ["price", "latitude", "longitude"], axis=1)
# Target vector y: Using the 'price' column as the target variable
y_train_districts = df_train_test["price"]

# Fit the model to the training data with districts and without latitude and longitude
rf_regressor_districts.fit(X_train_districts, y_train_districts)

# Predict on the final validation set with districts
y_pred_rf_districts = rf_regressor_districts.predict(
    df_final_val.drop(["price", "latitude", "longitude"], axis=1))

# If your target variable was log-transformed, apply the inverse transformation
y_pred_rf_districts_exp = np.exp(y_pred_rf_districts)
y_final_val_exp_districts = np.exp(df_final_val["price"])

# Calculate performance metrics
mse_rf_districts = mean_squared_error(
    y_final_val_exp_districts, y_pred_rf_districts_exp)
print("Random Forest Test MSE (with Districts, without Latitude and Longitude):", mse_rf_districts)


# Data transformation - Applying the inverse log transformation to 'area' and 'price'
df["area"] = df["area"].apply(lambda area_: np.exp(area_))
df["price"] = df["price"].apply(lambda price_: np.exp(price_))

# Concatenate the original unscaled features with the encoded district features
df_original_encoded = pd.concat([df, encoded_data_df], axis=1)

# Leaving 10% of the data for final evaluation of the model
df_train_test_original, df_final_val_original = train_test_split(
    df_original_encoded, test_size=0.1, random_state=42)

# Initialize the Random Forest regressor with default parameters
rf_regressor_original = RandomForestRegressor(random_state=42, n_jobs=-1)

# Features matrix X: Dropping the target variable 'price'
X_train_original = df_train_test_original.drop(["price"], axis=1)
# Target vector y: Using the 'price' column as the target variable
y_train_original = df_train_test_original["price"]

# Fit the model to the training data with original features and encoded districts
rf_regressor_original.fit(X_train_original, y_train_original)

# Predict on the final validation set with original features and encoded districts
y_pred_rf_original = rf_regressor_original.predict(
    df_final_val_original.drop(["price"], axis=1))

# If your target variable was log-transformed, apply the inverse transformation
y_pred_rf_original_exp = y_pred_rf_original
y_final_val_exp_original = df_final_val_original["price"]

# Calculate performance metrics
mse_rf_original = mean_squared_error(
    y_final_val_exp_original, y_pred_rf_original_exp)
# Calculate performance metrics
mape_rf_original = mean_absolute_percentage_error(
    y_final_val_exp_original, y_pred_rf_original_exp)
print("Random Forest Test MSE (with Original Features and Districts):", mse_rf_original)
print("Random Forest Test MAPE (with Original Features and Districts):", mape_rf_original)


# Specify the file path where you want to save the model
model_filename = "random_forest_model.pkl"

# Save the model to the specified file path
joblib.dump(rf_regressor_original, model_filename)

# Print a message to confirm that the model has been saved
print(f"Random Forest model saved to {model_filename}")
