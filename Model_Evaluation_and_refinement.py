import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import requests
from ipywidgets import interact, interactive, fixed, interact_manual
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


# Step 1: Download the file
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
response = requests.get(url)

# Step 2: Save the file locally
if response.status_code == 200:
    with open('module_5_auto.csv', 'wb') as file:
        file.write(response.content)
    print("File downloaded and saved as 'module_5_auto.csv'")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")

# Step 3: Load the dataset with pandas
df = pd.read_csv('module_5_auto.csv')
print("Original DataFrame:")
print(df.head())

# Step 4: Filter to only numeric data
df = df._get_numeric_data()  # Keep only numeric columns

# Step 5: Drop unnecessary columns
df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)  # Drop the specified columns if they exist

# Step 6: Display the updated DataFrame
print("Updated DataFrame:")
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_colwidth', None) 
print(df.head())

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    
    
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.legend()
    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    xmax = max([xtrain.values.max(), xtest.values.max()])
    xmin = min([xtrain.values.min(), xtest.values.min()])
    
    x = np.arange(xmin, xmax, 0.1)
    
    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    
    plt.ylim([-10000, 60000])
    plt.xlabel('Feature Value')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


y_data = df['price']

x_data=df.drop('price',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.40, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

lre=LinearRegression()

lre.fit(x_train[['horsepower']], y_train)

print(lre.score(x_test[['horsepower']], y_test))

print(lre.score(x_train[['horsepower']], y_train))

Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)

Rcross

print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')