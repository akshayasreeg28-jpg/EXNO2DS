# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("titanic_dataset.csv")
print("\nDataset Loaded Successfully\n")
print(data.head())
<img width="792" height="557" alt="image" src="https://github.com/user-attachments/assets/ea337dfe-ea5f-4a9a-8c49-27c380869336" />

print("\nDataset Info:\n")
print(data.info())
<img width="445" height="550" alt="image" src="https://github.com/user-attachments/assets/926eb376-c891-4680-b119-dc2713cd3457" />

print(data.describe())
<img width="702" height="526" alt="image" src="https://github.com/user-attachments/assets/9d33be92-48a8-4de9-b9e8-b82012c51c92" />

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])   # Mode for categorical
    else:
        data[column] = data[column].fillna(data[column].median())   # Median for numerical

print("\nMissing values handled successfully.\n")
plt.figure(figsize=(6,4))
sns.boxplot(x=data["Age"])
plt.title("Boxplot - Age")
plt.show()
<img width="675" height="558" alt="image" src="https://github.com/user-attachments/assets/61803ce0-8098-4737-b58f-7431c481d107" />

plt.figure(figsize=(6,4))
sns.boxplot(x=data["Fare"])
plt.title("Boxplot - Fare")
plt.show()
<img width="636" height="530" alt="image" src="https://github.com/user-attachments/assets/a4f82043-d4c7-46bc-a2e0-31fa937cdd1c" />

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

data = remove_outliers_iqr(data, "Age")
data = remove_outliers_iqr(data, "Fare")

print("Outliers removed using IQR method.\n")
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=data)
plt.title("Countplot - Survival Distribution")
plt.show()
<img width="728" height="527" alt="image" src="https://github.com/user-attachments/assets/02a1b244-f0ac-4074-9a88-15677cf18e8b" />


plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=data)
plt.title("Countplot - Gender Distribution")
plt.show()
<img width="702" height="528" alt="image" src="https://github.com/user-attachments/assets/49eadc8e-53a5-41c8-b8eb-df529323bf5a" />

plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=data)
plt.title("Countplot - Passenger Class Distribution")
plt.show()
<img width="728" height="530" alt="image" src="https://github.com/user-attachments/assets/82fa6a9e-f70c-43f2-9d3c-74536534d79e" />

sns.displot(data["Age"], kde=True, height=4, aspect=1.5)
plt.title("Displot - Age Distribution")
plt.show()
<img width="780" height="553" alt="image" src="https://github.com/user-attachments/assets/6b0dcf06-4fd8-4d87-8c86-dd6bbc143a87" />

sns.displot(data["Fare"], kde=True, height=4, aspect=1.5)
plt.title("Displot - Fare Distribution")
plt.show()
<img width="772" height="545" alt="image" src="https://github.com/user-attachments/assets/6c326f59-df83-4af3-9f54-e4fed24d3a43" />

print("\nCross Tabulation: Sex vs Survived\n")
print(pd.crosstab(data["Sex"], data["Survived"]))
<img width="387" height="368" alt="image" src="https://github.com/user-attachments/assets/bfe51e8f-5b86-4dc4-8972-31f787b524ed" />


print("\nCross Tabulation: Pclass vs Survived\n")
print(pd.crosstab(data["Pclass"], data["Survived"]))
plt.figure(figsize=(8,6))
correlation_matrix = data.select_dtypes(include=np.number).corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Titanic Dataset")
plt.show()
<img width="895" height="757" alt="image" src="https://github.com/user-attachments/assets/d05c20a9-d0af-4a00-8d4a-5289317221c1" />


# RESULT
Thus to perform Exploratory Data Analysis on the given data set is implemented.
