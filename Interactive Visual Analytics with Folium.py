import folium
import pandas as pd
# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon


## Task 1: Mark all launch sites on a map

# Download and read the `spacex_launch_geo.csv`

import io

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'

spacex_df=pd.read_csv(URL)
print(spacex_df.head(5))

# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
print(launch_sites_df.head())

#first need to create a folium Map object, with an initial center location to be NASA Johnson Space 
# Center at Houston, Texas.
# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)

#use folium.Circle to add a highlighted circle area with a text label on a specific coordinate.
# Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)

#TODO: Create and add folium.Circle and folium.Marker for each launch site on the site map
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

for index, row in launch_sites_df.iterrows():
    coordinate = [row['Lat'], row['Long']]
    
    folium.Circle(
        location=coordinate,
        radius=1000,  # Ajustar el radio del círculo
        color='#000000',  # Color negro
        fill=True,
        fill_color='#3186cc'
    ).add_child(folium.Popup(row['Launch Site'])).add_to(site_map)
    
    folium.map.Marker(
        location=coordinate,
        icon=DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html=f'<div style="font-size: 12px; color:#d35400;"><b>{row["Launch Site"]}</b></div>'
        )
    ).add_to(site_map)

site_map

# Task 2: Mark the success/failed launches for each site on the map
# Inicializar el mapa centrado en NASA Johnson Space Center
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

# Definir los colores para éxito y fallo
colors = {1: 'green', 0: 'red'}

# Crear un clúster para agrupar los marcadores
marker_cluster = MarkerCluster().add_to(site_map)

# Iterar sobre cada lanzamiento en el DataFrame original `spacex_df`
for index, row in spacex_df.iterrows():
    coordinate = [row['Lat'], row['Long']]
    launch_outcome = "Success" if row["class"] == 1 else "Failure"
    
    # Agregar marcador al clúster con el color correspondiente
    folium.Marker(
        location=coordinate,
        icon=folium.Icon(color=colors[row["class"]], icon="info-sign"),
        popup=f"{row['Launch Site']} - {launch_outcome}"
    ).add_to(marker_cluster)

# Mostrar el mapa
site_map

print(spacex_df.tail(10))

# TASK 3: Calculate the distances between a launch site to its proximities
# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

site_map.add_child(mouse_position)
site_map

from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance











































































