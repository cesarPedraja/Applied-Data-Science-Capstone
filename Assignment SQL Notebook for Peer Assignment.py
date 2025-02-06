import csv, sqlite3
import prettytable
prettytable.DEFAULT = 'DEFAULT'

con = sqlite3.connect("my_data1.db")
cur = con.cursor()

import pandas as pd
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")

#DROP THE TABLE IF EXISTS

#Task 1 Display the names of the unique launch sites in the space mission
query = 'SELECT DISTINCT "Launch_Site" FROM SPACEXTBL;'
df = pd.read_sql_query(query, con)
print(df)

#Task 2 Display 5 records where launch sites begin with the string 'CCA'
query = 'SELECT * FROM SPACEXTBL WHERE "Launch_Site" LIKE "CCA%" LIMIT 5;'
df = pd.read_sql_query(query, con)
print(df)

#Task 3 Display the total payload mass carried by boosters launched by NASA (CRS)
query =  'SELECT SUM ("PAYLOAD_MASS__KG_") FROM SPACEXTBL WHERE "Customer" = "NASA (CRS)";'
df = pd.read_sql_query(query, con)
print(df)

#Task 4 Display average payload mass carried by booster version F9 v1.1
query =  'SELECT AVG ("PAYLOAD_MASS__KG_") FROM SPACEXTBL WHERE "Booster_Version" LIKE "F9 v1.1";'
df = pd.read_sql_query(query, con)
print(df)

#Task 5 List the date when the first succesful landing outcome in ground pad was acheived.
query =  'SELECT * FROM SPACEXTBL WHERE "Landing_Outcome" = "Success (ground pad)" ORDER BY "Date" ASC LIMIT 1;'
df = pd.read_sql_query(query, con)
print(df)

#Task 6 List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000
query = '''SELECT "Booster_Version" FROM SPACEXTBL WHERE "Landing_Outcome" = 'Success (drone ship)' AND "PAYLOAD_MASS__KG_" BETWEEN 4000 AND 6000;'''
df = pd.read_sql_query(query, con)
print(df)

#Task 7 List the total number of successful and failure mission outcomes
query = '''SELECT "Mission_Outcome", COUNT(*) as Total FROM SPACEXTBL GROUP BY "Mission_Outcome";'''
df = pd.read_sql_query(query, con)
print(df)

#Task 8 List the names of the booster_versions which have carried the maximum payload mass. Use a subquery
query = '''SELECT "Booster_Version" FROM SPACEXTBL WHERE "PAYLOAD_MASS__KG_" = (SELECT MAX("PAYLOAD_MASS__KG_") FROM SPACEXTBL);'''
df = pd.read_sql_query(query, con)
print(df)


#Task 9
#List the records which will display the month names, failure landing_outcomes in drone ship ,booster versions, launch_site for the months in year 2015.
#Note: SQLLite does not support monthnames. So you need to use substr(Date, 6,2) as month to get the months and substr(Date,0,5)='2015' for year.
query = '''SELECT substr("Date", 6,2) as Month, "Landing_Outcome", "Booster_Version", "Launch_Site" FROM SPACEXTBL WHERE substr("Date",0,5)='2015' AND "Landing_Outcome" LIKE 'Failure (%drone ship%)';'''
df = pd.read_sql_query(query, con)
print(df)

#Task 10 Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order.
query = '''SELECT "Landing_Outcome", COUNT(*) as Total FROM SPACEXTBL WHERE "Date" BETWEEN '2010-06-04' AND '2017-03-20' GROUP BY "Landing_Outcome" ORDER BY Total DESC;'''
df = pd.read_sql_query(query, con)
print(df)




















































