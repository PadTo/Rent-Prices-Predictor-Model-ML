import requests
from key import *
from constants import *
from PIL import Image
from io import BytesIO


ACCESS_TOKEN = key_mapbox

# Define the bounding box with your coordinates
# [long_west, lat_south, long_east, lat_north]
bbox = [min_lon, min_lat, max_lon, max_lat]

url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/static/{bbox}/{width}x{height}@2x?access_token={ACCESS_TOKEN}"

response = requests.get(url=url)
# Check if the request was successful
if response.status_code == 200:
    # Load the image
    map_image = Image.open(BytesIO(response.content))
    map_image.save('datasets/mapbox_image.png')
    print("Map image saved successfully.")

else:
    print(
        f"Failed to retrieve the map image. {response.status_code} {response.content}")
