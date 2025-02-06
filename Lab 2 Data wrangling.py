#Import Libraries and Define Auxiliary Functions
# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np

#Data Analysis

df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
print(df.head(10))

#Identify and calculate the percentage of the missing values in each attribute
print(df.isnull().sum()/len(df)*100)

#Identify which columns are numerical and categorical:
print(df.dtypes)

#TASK 1: Calculate the number of launches on each site
print(df['LaunchSite'].value_counts())

#TASK 2: Calculate the number and occurrence of each orbit
print(df['Orbit'].value_counts())

#TASK 3: Calculate the number and occurence of mission outcome of the orbits
landing_outcomes = (df['Outcome'].value_counts())
print(landing_outcomes)

for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)

#We create a set of outcomes where the second stage did not land successfully:

bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes

#TASK 4: Create a landing outcome label from Outcome column
landing_class = [0 if outcome in bad_outcomes else 1 for outcome in df['Outcome']]

#This variable will represent the classification variable that represents the outcome of each launch. 
#If the value is zero, the first stage did not land successfully; one means the first stage landed Successfully

df['Class']=landing_class
print(df[['Class']].head(8))

print(df.head(5))

#We can use the following line of code to determine the success rate
print(df["Class"].mean())

df.to_csv("dataset_part_2.csv", index=False)


































































