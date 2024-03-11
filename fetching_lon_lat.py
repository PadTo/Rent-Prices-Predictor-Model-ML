
import pandas as pd
import requests
from key import *
import time


df = pd.read_excel("Vilnius_flats_for_rent_fixed.xlsx")
url = "https://eu1.locationiq.com/v1/search"


def collect_into_param_structure(address, country="Lithuania"):
    data_temp = {
        'key': key,
        'street': address,
        'country': country,
        'format': 'json'
    }

    return data_temp


lon_col = []
lat_col = []

for index in df.index:
    address = df["address"][index]

    parameters = collect_into_param_structure(address=address)
    response = requests.get(url=url, params=parameters)

    if response.status_code == 200:
        data = response.json()
        if data and 'lat' in data[0] and 'lon' in data[0]:
            longitude = float(data[0]["lon"])
            latitude = float(data[0]["lat"])

            df.at[index, "longitude"] = longitude
            df.at[index, "latitude"] = latitude
            time.sleep(0.6)
            print(f"Successful, index: {index}")

    else:
        print("error: " + str(response.status_code))


df.to_csv(r'C:\Users\tpade\Desktop\Vilnius_flats_for_rent_fixed_with_coordinates.csv', index=False)
