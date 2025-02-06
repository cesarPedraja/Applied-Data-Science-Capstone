SpaceX Launch Data Analysis
This project retrieves, processes, and analyzes past SpaceX launch data using the SpaceX API. The goal is to extract relevant launch details, such as rocket type, payload mass, launch site coordinates, and mission outcomes, while filtering the dataset to include only Falcon 9 launches.

Features
API Integration: Uses requests to fetch SpaceX launch data.
Data Cleaning & Transformation:
Extracts booster version, launch site, payload details, and core information.
Converts JSON responses into a structured pandas DataFrame.
Filters data to include only Falcon 9 launches.
Handles missing values by replacing them with the mean of relevant columns.
Export: Saves the cleaned dataset as dataset_part_1.csv.
Technologies Used
Python
requests (API calls)
pandas (data processing)
numpy (handling missing values)
datetime (date transformations)
How to Use
Run the script to fetch SpaceX launch data from the API.
Process and clean the data to extract relevant information.
Export the structured dataset for further analysis.
This dataset can be used for further predictive analysis, visualization, or machine learning applications related to SpaceX launches. 🚀
