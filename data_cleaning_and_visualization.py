import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from key import *
from constants import *
from PIL import Image

# Data Cleaning and Manipulation

# Analyzing data
df = pd.read_csv("datasets/Vilnius_flats_for_rent_fixed_with_coordinates.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df["District"].value_counts().count())
print(df["Street"].value_counts().count())

# Converting floor type to string from object
df["floor"] = df["floor"].astype(str)

# Adding building size in floors
df["building size (floors)"] = df["floor"].apply(
    lambda x: int(x.split('/')[1]))

# Cleaning the floor by taking into acount only the floor number and not the floor number with respect to max floors in the building
df["floor"] = df["floor"].apply(lambda x:  int(x.split('/')[0]))


df_ml = df.drop(["address", "Street", "Country",
                "District"], axis=1, inplace=False)
print(df_ml.corr())


def plot_formatted(data_frame):
    data_frame.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                    s=data_frame["area"]/8, label="Apartment Area",
                    c="price", colormap="jet", colorbar=True,
                    legend=True, sharex=True, figsize=(10, 7), alpha=0.8)


# Visualizing data
df_ml.drop(["index"], axis=1, inplace=False).hist(
    bins=50, figsize=(14, 8))
plt.show()


plot_formatted(df_ml)
plt.show()
plt.close()


filtered_df = df_ml[
    (df_ml["longitude"] >= min_lon) & (df_ml["longitude"] <= max_lon) &
    (df_ml["latitude"] >= min_lat) & (df_ml["latitude"] <= max_lat)]

print(filtered_df["index"].count())


plot_formatted(filtered_df)
plt.show()


# Load the saved image
map_image_path = 'datasets/mapbox_image.png'
map_image = Image.open(map_image_path)

# Get the dimensions of the image for plotting
width, height = map_image.size

# Set up the plot figure
fig, ax = plt.subplots(figsize=(10, 7))

# Display the map image
ax.imshow(map_image, extent=[min_lon, max_lon, min_lat, max_lat])

# Plot the scatter plot on top of the map image
sc = ax.scatter(filtered_df['longitude'], filtered_df['latitude'],
                c=filtered_df['price'], s=filtered_df['area'] /
                8,  # Adjust the size as needed
                alpha=0.8, cmap='jet')

# Add a colorbar
colorbar = fig.colorbar(sc, ax=ax)
colorbar.set_label('Price')

# Set the x and y labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Optionally set the title
ax.set_title('Apartments for Rent in Vilnius')

# Show the plot
plt.show()
