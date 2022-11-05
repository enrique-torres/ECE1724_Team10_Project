import json
from pathlib import Path
import csv


def nested_lists(coordinates_info, level):
    if type(coordinates_info) == list:
        return nested_lists(coordinates_info[0], level + 1)
    else:
        return level

def denest_list(coordinates_info):
    level = max(0, nested_lists(coordinates_info, 0) - 2)
    returned_list = coordinates_info
    for i in range(level):
        returned_list = returned_list[0]
    return returned_list


with open(Path("./Datasets/shops.geojson"), mode='r', encoding='utf-8') as shops_file:
    shops_json = json.load(shops_file)
    with open('shops.csv', mode='w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['ID', 'Latitude', 'Longitude', 'Shop Weight'])
        current_shop_id = 0
        for feature in shops_json['features']:
            shop_lats = []
            shop_longs = []
            shop_lat = None
            shop_lon = None
            shop_weight = None

            if feature['geometry']['type'] == "MultiPolygon" or feature['geometry']['type'] == "Polygon" or feature['geometry']['type'] == "LineString":
                denested_list = denest_list(feature['geometry']['coordinates'])
                for coordinate in denested_list:
                    shop_lats.append(coordinate[0])
                    shop_longs.append(coordinate[1])

                shop_lat = sum(shop_lats) / len(shop_lats)
                shop_lon = sum(shop_longs) / len(shop_longs)
                shop_weight = len(shop_lats) # give it a weight based on the multipolygon number of corners, to keep in mind big shops
                print("Added polygon shop with (lat,lon): (" + str(shop_lat) + ", " + str(shop_lon) + ") and ID: " + str(feature['id']))
            elif feature['geometry']['type'] == "Point":
                shop_lat = feature['geometry']['coordinates'][0]
                shop_lon = feature['geometry']['coordinates'][1]
                shop_weight = 1
                print("Added point shop with (lat,lon): (" + str(shop_lat) + ", " + str(shop_lon) + ") and ID: " + str(feature['id']))
            else:
                print("Unknown feature type, found type is: " + str(feature['geometry']['type']))
            
            if shop_lat is not None and shop_lon is not None and shop_weight is not None:
                csvwriter.writerow([current_shop_id, shop_lat, shop_lon, shop_weight])