# A Machine Learning Project on Predicting the price of the laptop given its features

## Data Cleaning, Preprocessing and Basic Data Analysis

Machine Learning or Data Analysis is performed on the data obtained from the various interactions happening around us.

Most of the times the data we collect is not pure and is instead a combination of categorical, numerical, Time-series or text data with a few inconsistent values here and there. We as humans have evolved enough to understand data in various forms and are able to establish relationships amongst them easily but then machines are not so evolved and so are unable to understand anything other than numerical data, and so we are forced to clean the data to remove inconsistencies and apply various preprocessing techniques to reduce the data into a form that is understandable by the machine.

Data cleaning or analysis and preprocessing do not tend to follow a hard fast rule of approach and is instead dependent on the type of dataset being studied, i.e approach varies for each dataset. 

So always have a habit of going throught the dataset, it's meaning and significance in the real world before starting any approach.

Do tailor your approach to the dataset in hand so that proper sense can be made from the dataset.

### Importing the required libraries, files, performing basic analysis and cleaning the data


```python
# importing the required libraries

import pandas as pd
pd.set_option("display.precision", 2)
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
```


```python
# importing and reading the .csv file

df = pd.read_csv('laptop_price.csv')
print("The number of rows are", df.shape[0],"and the number of columns are", df.shape[1])
df.head()
```

    The number of rows are 1303 and the number of columns are 13
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>laptop_ID</th>
      <th>Company</th>
      <th>Product</th>
      <th>TypeName</th>
      <th>Inches</th>
      <th>ScreenResolution</th>
      <th>Cpu</th>
      <th>Ram</th>
      <th>Memory</th>
      <th>Gpu</th>
      <th>OpSys</th>
      <th>Weight</th>
      <th>Price_euros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>IPS Panel Retina Display 2560x1600</td>
      <td>Intel Core i5 2.3GHz</td>
      <td>8GB</td>
      <td>128GB SSD</td>
      <td>Intel Iris Plus Graphics 640</td>
      <td>macOS</td>
      <td>1.37kg</td>
      <td>1339.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Apple</td>
      <td>Macbook Air</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>1440x900</td>
      <td>Intel Core i5 1.8GHz</td>
      <td>8GB</td>
      <td>128GB Flash Storage</td>
      <td>Intel HD Graphics 6000</td>
      <td>macOS</td>
      <td>1.34kg</td>
      <td>898.94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>HP</td>
      <td>250 G6</td>
      <td>Notebook</td>
      <td>15.6</td>
      <td>Full HD 1920x1080</td>
      <td>Intel Core i5 7200U 2.5GHz</td>
      <td>8GB</td>
      <td>256GB SSD</td>
      <td>Intel HD Graphics 620</td>
      <td>No OS</td>
      <td>1.86kg</td>
      <td>575.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>15.4</td>
      <td>IPS Panel Retina Display 2880x1800</td>
      <td>Intel Core i7 2.7GHz</td>
      <td>16GB</td>
      <td>512GB SSD</td>
      <td>AMD Radeon Pro 455</td>
      <td>macOS</td>
      <td>1.83kg</td>
      <td>2537.45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>IPS Panel Retina Display 2560x1600</td>
      <td>Intel Core i5 3.1GHz</td>
      <td>8GB</td>
      <td>256GB SSD</td>
      <td>Intel Iris Plus Graphics 650</td>
      <td>macOS</td>
      <td>1.37kg</td>
      <td>1803.60</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the information of the dataframe(i.e the dataset)

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1303 entries, 0 to 1302
    Data columns (total 13 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   laptop_ID         1303 non-null   int64  
     1   Company           1303 non-null   object 
     2   Product           1303 non-null   object 
     3   TypeName          1303 non-null   object 
     4   Inches            1303 non-null   float64
     5   ScreenResolution  1303 non-null   object 
     6   Cpu               1303 non-null   object 
     7   Ram               1303 non-null   object 
     8   Memory            1303 non-null   object 
     9   Gpu               1303 non-null   object 
     10  OpSys             1303 non-null   object 
     11  Weight            1303 non-null   object 
     12  Price_euros       1303 non-null   float64
    dtypes: float64(2), int64(1), object(10)
    memory usage: 132.5+ KB
    


```python
# Seeing the description of the dataframe

df.describe(include=['object', 'int64', 'float64'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>laptop_ID</th>
      <th>Company</th>
      <th>Product</th>
      <th>TypeName</th>
      <th>Inches</th>
      <th>ScreenResolution</th>
      <th>Cpu</th>
      <th>Ram</th>
      <th>Memory</th>
      <th>Gpu</th>
      <th>OpSys</th>
      <th>Weight</th>
      <th>Price_euros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1303.00</td>
      <td>1303</td>
      <td>1303</td>
      <td>1303</td>
      <td>1303.00</td>
      <td>1303</td>
      <td>1303</td>
      <td>1303</td>
      <td>1303</td>
      <td>1303</td>
      <td>1303</td>
      <td>1303</td>
      <td>1303.00</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>19</td>
      <td>618</td>
      <td>6</td>
      <td>NaN</td>
      <td>40</td>
      <td>118</td>
      <td>9</td>
      <td>40</td>
      <td>110</td>
      <td>9</td>
      <td>179</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>Dell</td>
      <td>XPS 13</td>
      <td>Notebook</td>
      <td>NaN</td>
      <td>Full HD 1920x1080</td>
      <td>Intel Core i5 7200U 2.5GHz</td>
      <td>8GB</td>
      <td>256GB SSD</td>
      <td>Intel HD Graphics 620</td>
      <td>Windows 10</td>
      <td>2.2kg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>297</td>
      <td>30</td>
      <td>727</td>
      <td>NaN</td>
      <td>507</td>
      <td>190</td>
      <td>619</td>
      <td>412</td>
      <td>281</td>
      <td>1072</td>
      <td>121</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>660.16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1123.69</td>
    </tr>
    <tr>
      <th>std</th>
      <td>381.17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.43</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>699.01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>174.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>331.50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>599.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>659.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>977.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>990.50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1487.88</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1320.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.40</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6099.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking all the different unique values

df.nunique()
```




    laptop_ID           1303
    Company               19
    Product              618
    TypeName               6
    Inches                18
    ScreenResolution      40
    Cpu                  118
    Ram                    9
    Memory                40
    Gpu                  110
    OpSys                  9
    Weight               179
    Price_euros          791
    dtype: int64



The reason why we see the number of unique values for each column is to help us decide on type of encoding to use inorder to convert the catergorical values into a form understandable by the machine. 

For example generally label encoding is used if there are too many categorical values and one-hot encoding if there are less categorical values, the reason for which lies in the means each method encodes the values.

One-hot encoding tends to create one less than the number of unique categories (to prevent dummy variable trap) available in the column and more the unique values more the columns and more the chance of overfitting the model while label encoding tends to label each category based on the amount of importance it has, which works best inside a column and begins affecting the accuracy, incase you use it for many columns.

So a combination of such encoding techniques would have to be used to obtain a model with really good accuracy.


```python
# Checking the number of null/missing values in the dataframe

print(df.isnull().sum())
```

    laptop_ID           0
    Company             0
    Product             0
    TypeName            0
    Inches              0
    ScreenResolution    0
    Cpu                 0
    Ram                 0
    Memory              0
    Gpu                 0
    OpSys               0
    Weight              0
    Price_euros         0
    dtype: int64
    

This is done to see the effectiveness of the dataset, more missing values means the dataset is a bad one. 

If incase there are little missing values we simple replace them with mean, median or many other such values after selecting the best one.


```python
# Checking the number of 0 values in the dataframe

df = df.replace(0.0, np.nan)
df.isnull().sum()
```




    laptop_ID           0
    Company             0
    Product             0
    TypeName            0
    Inches              0
    ScreenResolution    0
    Cpu                 0
    Ram                 0
    Memory              0
    Gpu                 0
    OpSys               0
    Weight              0
    Price_euros         0
    dtype: int64



This is not necessary but then after a through study of Laptop Sales data, we can tell with confidence that no value is supposed to be 0 and so we perform this check.

### Now let's go through each and every column, performing the proper preprocessing techniques and analysis of the column

#### Let's start with the Company of the Laptops


```python
# Here we see all the companies who have produced Laptops

df['Company'].unique()
```




    array(['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI',
           'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer',
           'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG'], dtype=object)




```python
# Here we see the number of laptops sold by each company

plt.title("Market Share of each Laptop Manufacturer", fontsize=20)
df['Company'].value_counts().plot(kind = 'bar', rot = 45, sort_columns = True, figsize = (20,10), fontsize = 15)
df['Company'].value_counts()
```




    Dell         297
    Lenovo       297
    HP           274
    Asus         158
    Acer         103
    MSI           54
    Toshiba       48
    Apple         21
    Samsung        9
    Razer          7
    Mediacom       7
    Microsoft      6
    Vero           4
    Xiaomi         4
    Chuwi          3
    Google         3
    LG             3
    Fujitsu        3
    Huawei         2
    Name: Company, dtype: int64




    
![png](output_17_1.png)
    



```python
# Same as that of before but as a percentage pie-chart

plt.title("Market Share of each Laptop Manufacturer", fontsize=20)
df['Company'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize = (20,15), fontsize = 12).set_ylabel('')
df['Company'].value_counts(normalize = True)*100
```




    Dell         22.79
    Lenovo       22.79
    HP           21.03
    Asus         12.13
    Acer          7.90
    MSI           4.14
    Toshiba       3.68
    Apple         1.61
    Samsung       0.69
    Razer         0.54
    Mediacom      0.54
    Microsoft     0.46
    Vero          0.31
    Xiaomi        0.31
    Chuwi         0.23
    Google        0.23
    LG            0.23
    Fujitsu       0.23
    Huawei        0.15
    Name: Company, dtype: float64




    
![png](output_18_1.png)
    


From the above we can see that Dell and Lenovo have the most share in the market followed by HP, Asus, Acer and so on.
This shows that Dell and Lenovo have a good brand value and a loyal customer base, this information is helpful to merchants so that they can stock more laptops made by Dell and Lenovo.

#### Now let's move on to the Laptop Types


```python
# Here we see all the different types of Laptops available

df['TypeName'].unique()
```




    array(['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible',
           'Workstation'], dtype=object)




```python
# Here we see the number of laptops of each type sold

plt.title("Market Share of each Laptop Type", fontsize=20)
df['TypeName'].value_counts().plot(kind = 'bar', rot = 45, sort_columns = True, figsize = (15,10), fontsize = 12)
df['TypeName'].value_counts()
```




    Notebook              727
    Gaming                205
    Ultrabook             196
    2 in 1 Convertible    121
    Workstation            29
    Netbook                25
    Name: TypeName, dtype: int64




    
![png](output_22_1.png)
    



```python
# Same as that of before but as a percentage pie-chart

plt.title("Market Share of each Laptop Type", fontsize=20)
df['TypeName'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize = (15,10), fontsize = 12).set_ylabel('')
df['TypeName'].value_counts(normalize = True)*100
```




    Notebook              55.79
    Gaming                15.73
    Ultrabook             15.04
    2 in 1 Convertible     9.29
    Workstation            2.23
    Netbook                1.92
    Name: TypeName, dtype: float64




    
![png](output_23_1.png)
    


We can see the Notebook Laptops dominate the market by more than a 55% share followed by Gaming Laptops, Ultrabook and so on. This gives us the information that people tend to prefer Laptop Notebooks over other types and we can also see that Netbooks have the lowest share. 

#### Now let's see the Laptop sizes


```python
# Here we see all the different sizes of Laptops available

df['Inches'].unique()
```




    array([13.3, 15.6, 15.4, 14. , 12. , 11.6, 17.3, 10.1, 13.5, 12.5, 13. ,
           18.4, 13.9, 12.3, 17. , 15. , 14.1, 11.3])




```python
# Here we see the number of laptops of each size sold

df['Inches'] = pd.to_numeric(df['Inches'])
plt.title("Market Share of each Laptop based on Size", fontsize=20)
df['Inches'].value_counts().plot(kind = 'bar', rot = 0, sort_columns = True, figsize = (15,10), fontsize = 12)
df['Inches'].value_counts()
```




    15.6    665
    14.0    197
    13.3    164
    17.3    164
    12.5     39
    11.6     33
    13.5      6
    13.9      6
    12.0      6
    12.3      5
    10.1      4
    15.0      4
    15.4      4
    13.0      2
    11.3      1
    14.1      1
    17.0      1
    18.4      1
    Name: Inches, dtype: int64




    
![png](output_27_1.png)
    



```python
# Same as that of before but as a percentage pie-chart

plt.title("Market Share of each Laptop based on Size", fontsize=20)
df['Inches'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize = (15,10), fontsize = 12).set_ylabel('')
df['Inches'].value_counts(normalize = True)*100
```




    15.6    51.04
    14.0    15.12
    13.3    12.59
    17.3    12.59
    12.5     2.99
    11.6     2.53
    13.5     0.46
    13.9     0.46
    12.0     0.46
    12.3     0.38
    10.1     0.31
    15.0     0.31
    15.4     0.31
    13.0     0.15
    11.3     0.08
    14.1     0.08
    17.0     0.08
    18.4     0.08
    Name: Inches, dtype: float64




    
![png](output_28_1.png)
    


We can see that many people prefer a laptop size of 15.6 inches which is not so huge and yet not so small too. That is followed by 14.0 inches and so on with very less people prefering 11.3 (a little big than the smallest size available in the market) and 18.4 (the largest available size)

#### Now let us look at the Screen Resolution of the laptops

The column ScreenResolution provides an interesting problem, the data in the column is a string but also consists of a numerical data which is the screen width and height, even the data that tells us about whether the laptop has a touchscreen, OLD screen, IPS Panel or a normal screen, etc., is hidden in this column. We, on going through the values, can say what is what but a machine, unfortunately, cannot and so we must properly divide it into its individual constituents, so as to get the data into a format that can be understood by a machine.

That is exactly what we intend to do here, we first split the screen size from the data, put it into a separate column, and then we further split the screen size into its constituent width and height(effectively converting it into a numerical data).

We then go ahead by creating a separate column to specify if the laptop has a touchscreen or not.

So all in all we are splitting one column into 4 different columns thereby making the data more understandable and usable in training a model.


```python
# Here we are splitting the column of ScreenResolution into their constituent types

# First is the Screen Size

Cpu = df['ScreenResolution'].str.split()
Screen_Size = Cpu.str.get(-1)
df.insert(5, 'Screen Size', Screen_Size)
Screen_Type = []
for i in range(df.shape[0]):
    Cpu[i].remove(Screen_Size[i])
    Screen_Type.append(' '.join(Cpu[i]))
df['ScreenResolution'] = Screen_Type

# Now we are splitting the screen size into screen width and screen height

Screen = df['Screen Size'].str.split('x', expand=True)
df.insert(5, 'Screen Width', Screen[0])
df['Screen Size'] = Screen[1]
df.rename(columns={'Screen Size':'Screen Height'}, inplace=True)
df['Screen Width'] = pd.to_numeric(df['Screen Width'])
df['Screen Height'] = pd.to_numeric(df['Screen Height'])

# Creating a column known as TouchScreen to signify whether the laptop has a touch screen.

Screen = df['ScreenResolution'].str.split(' / ')
TouchScreen = []
for i in range(len(Screen)):
    if 'Touchscreen' in Screen[i] or 'IPS Panel Touchscreen' in Screen[i] or 'IPS Panel Touchscreen 4K Ultra HD' in Screen[i]:
        TouchScreen.append('Yes')
        if 'IPS Panel Touchscreen' not in Screen[i]:
            del Screen[i][Screen[i].index('Touchscreen')]
    else:
        TouchScreen.append('No')
    Screen[i] = ' '.join(Screen[i])
df.insert(8, 'TouchScreen', TouchScreen)
df['ScreenResolution'] = Screen
df['ScreenResolution'] = df['ScreenResolution'].replace('','None')
df.rename(columns={'ScreenResolution':'Screen Type'}, inplace=True)
```


```python
# displaying the dataframe with all the analysis done as of now

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>laptop_ID</th>
      <th>Company</th>
      <th>Product</th>
      <th>TypeName</th>
      <th>Inches</th>
      <th>Screen Width</th>
      <th>Screen Height</th>
      <th>Screen Type</th>
      <th>TouchScreen</th>
      <th>Cpu</th>
      <th>Ram</th>
      <th>Memory</th>
      <th>Gpu</th>
      <th>OpSys</th>
      <th>Weight</th>
      <th>Price_euros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel Core i5 2.3GHz</td>
      <td>8GB</td>
      <td>128GB SSD</td>
      <td>Intel Iris Plus Graphics 640</td>
      <td>macOS</td>
      <td>1.37kg</td>
      <td>1339.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Apple</td>
      <td>Macbook Air</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>1440</td>
      <td>900</td>
      <td>None</td>
      <td>No</td>
      <td>Intel Core i5 1.8GHz</td>
      <td>8GB</td>
      <td>128GB Flash Storage</td>
      <td>Intel HD Graphics 6000</td>
      <td>macOS</td>
      <td>1.34kg</td>
      <td>898.94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>HP</td>
      <td>250 G6</td>
      <td>Notebook</td>
      <td>15.6</td>
      <td>1920</td>
      <td>1080</td>
      <td>Full HD</td>
      <td>No</td>
      <td>Intel Core i5 7200U 2.5GHz</td>
      <td>8GB</td>
      <td>256GB SSD</td>
      <td>Intel HD Graphics 620</td>
      <td>No OS</td>
      <td>1.86kg</td>
      <td>575.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>15.4</td>
      <td>2880</td>
      <td>1800</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel Core i7 2.7GHz</td>
      <td>16GB</td>
      <td>512GB SSD</td>
      <td>AMD Radeon Pro 455</td>
      <td>macOS</td>
      <td>1.83kg</td>
      <td>2537.45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel Core i5 3.1GHz</td>
      <td>8GB</td>
      <td>256GB SSD</td>
      <td>Intel Iris Plus Graphics 650</td>
      <td>macOS</td>
      <td>1.37kg</td>
      <td>1803.60</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Performing our analysis on the Laptop Screen Type

print(f"There are {len(df[df['Screen Type'] == 'None'])} laptops whose Screen Type is not available or is unspecified.")
print("So let's perform our analysis on those laptops that have their Screen Type properly mentioned")
Screen_withoutNone = df[df['Screen Type'] != 'None']
plt.title("Number of Laptops with a specific Screen Type", fontsize=20)
Screen_withoutNone['Screen Type'].value_counts().plot(kind = 'bar', rot = 60, sort_columns = True, figsize = (15,10), fontsize = 12)
Screen_withoutNone['Screen Type'].value_counts()
```

    There are 346 laptops whose Screen Type is not available or is unspecified.
    So let's perform our analysis on those laptops that have their Screen Type properly mentioned
    




    Full HD                              555
    IPS Panel Full HD                    288
    IPS Panel 4K Ultra HD                 23
    Quad HD+                              19
    4K Ultra HD                           18
    IPS Panel Retina Display              17
    IPS Panel Touchscreen                 13
    IPS Panel                             11
    IPS Panel Quad HD+                    11
    IPS Panel Touchscreen 4K Ultra HD      2
    Name: Screen Type, dtype: int64




    
![png](output_34_2.png)
    



```python
# Same as that of before but as a percentage pie-chart

plt.title("Number of Laptops with a specific Screen Type", fontsize=20)
Screen_withoutNone['Screen Type'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize = (15,10), fontsize = 12).set_ylabel('')
Screen_withoutNone['Screen Type'].value_counts(normalize = True)*100
```




    Full HD                              57.99
    IPS Panel Full HD                    30.09
    IPS Panel 4K Ultra HD                 2.40
    Quad HD+                              1.99
    4K Ultra HD                           1.88
    IPS Panel Retina Display              1.78
    IPS Panel Touchscreen                 1.36
    IPS Panel                             1.15
    IPS Panel Quad HD+                    1.15
    IPS Panel Touchscreen 4K Ultra HD     0.21
    Name: Screen Type, dtype: float64




    
![png](output_35_1.png)
    


We can see that laptops with Full HD screens, having a market share of around 58%, are more preferred by the people followed by laptops with IPS Panel Full HD screens, having a market share of around 30.1% over others with IPS Panel Touchscreen 4K Ultra HD being the least preferred, and this may be due to the huge cost of such laptops.


```python
# Displaying the number of laptops with and without TouchScreen

print('No of laptops without and with touchscreens')
print(df['TouchScreen'].value_counts())
plt.title("Number of laptops with or without Touchscreens", fontsize=20)
df['TouchScreen'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize = (15,10), fontsize = 12).set_ylabel('')
df['TouchScreen'].value_counts(normalize = True)*100
```

    No of laptops without and with touchscreens
    No     1111
    Yes     192
    Name: TouchScreen, dtype: int64
    




    No     85.26
    Yes    14.74
    Name: TouchScreen, dtype: float64




    
![png](output_37_2.png)
    


We can see that laptops without touchscreens dominate the market which means that people prefer laptops without touchscreen more than their counterparts. 

#### Now let us look at the CPU of the laptops

The column CPU also poses the same problem as the Screen Resolution and we follow the same methods as above to split the column into 5 different columns which is easier for the machine to understand.


```python
# Here we are splitting the column of CPU into their constituent types

# First is the Cpu Vendor(i.e company that made the chipset)

Cpu_vendor = df['Cpu'].str.split(' ', 1, expand=True)
df.insert(9, 'Cpu Vendor', Cpu_vendor[0])
df['Cpu'] = Cpu_vendor[1]

# Next is the Type of the Cpu

Cpu_Type = df['Cpu'].str.split(' ', 1, expand=True)
df.insert(10, 'Cpu Type', Cpu_Type[0])
df['Cpu'] = Cpu_Type[1]

# Then we seperate the Cpu Speed

Cpu = df['Cpu'].str.split()
Cpu_speed = Cpu.str.get(-1)
df.insert(12, 'Cpu Speed', Cpu_speed)
Cpu_processor = []
for i in range(df.shape[0]):
    Cpu[i].remove(Cpu_speed[i])
    Cpu_processor.append(' '.join(Cpu[i]))
df['Cpu'] = Cpu_processor
df['Cpu Speed'] = df['Cpu Speed'].replace('[GHz]', '', regex=True)
df['Cpu Speed'] = pd.to_numeric(df['Cpu Speed'])
df.rename(columns={'Cpu Speed':'Cpu Speed (GHz)'}, inplace=True)

# Finally we divide the model and series from the column and arrange them

Cpu = list(df['Cpu'].str.split())
Cpu_model = []
Cpu_series = []
for i in Cpu:
    if len(i) == 1:
        if len(i[0].split('-')) == 2:           # this is to accomodate the data which have a '-' between the model and series
            model = i[0].split('-')
            Cpu_series.append(str(model[0]))
            Cpu_model.append(str(model[1]))
        else:
            Cpu_series.append(i[0])
            if i[0] not in ['1600', '1700']:    # this is to accomodate the Ryzen CPU whose Cpu series is dependant on the Cpu model 
                Cpu_model.append('None')
            elif i[0] == '1600':
                Cpu_model.append('5')
            elif i[0] == '1700':
                Cpu_model.append('7')
    elif len(i) >= 2:
        Cpu_series.append(str(' '.join(i[:len(i)-1])))
        Cpu_model.append(str(i[-1]))
df.insert(11, 'Cpu Series', Cpu_series)
df['Cpu'] = Cpu_model
df.rename(columns={'Cpu':'Cpu Model'}, inplace=True)
```


```python
# Displaying the dataframe with all the analysis done as of now

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>laptop_ID</th>
      <th>Company</th>
      <th>Product</th>
      <th>TypeName</th>
      <th>Inches</th>
      <th>Screen Width</th>
      <th>Screen Height</th>
      <th>Screen Type</th>
      <th>TouchScreen</th>
      <th>Cpu Vendor</th>
      <th>Cpu Type</th>
      <th>Cpu Series</th>
      <th>Cpu Model</th>
      <th>Cpu Speed (GHz)</th>
      <th>Ram</th>
      <th>Memory</th>
      <th>Gpu</th>
      <th>OpSys</th>
      <th>Weight</th>
      <th>Price_euros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>2.3</td>
      <td>8GB</td>
      <td>128GB SSD</td>
      <td>Intel Iris Plus Graphics 640</td>
      <td>macOS</td>
      <td>1.37kg</td>
      <td>1339.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Apple</td>
      <td>Macbook Air</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>1440</td>
      <td>900</td>
      <td>None</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>1.8</td>
      <td>8GB</td>
      <td>128GB Flash Storage</td>
      <td>Intel HD Graphics 6000</td>
      <td>macOS</td>
      <td>1.34kg</td>
      <td>898.94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>HP</td>
      <td>250 G6</td>
      <td>Notebook</td>
      <td>15.6</td>
      <td>1920</td>
      <td>1080</td>
      <td>Full HD</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>7200U</td>
      <td>2.5</td>
      <td>8GB</td>
      <td>256GB SSD</td>
      <td>Intel HD Graphics 620</td>
      <td>No OS</td>
      <td>1.86kg</td>
      <td>575.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>15.4</td>
      <td>2880</td>
      <td>1800</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i7</td>
      <td>None</td>
      <td>2.7</td>
      <td>16GB</td>
      <td>512GB SSD</td>
      <td>AMD Radeon Pro 455</td>
      <td>macOS</td>
      <td>1.83kg</td>
      <td>2537.45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>3.1</td>
      <td>8GB</td>
      <td>256GB SSD</td>
      <td>Intel Iris Plus Graphics 650</td>
      <td>macOS</td>
      <td>1.37kg</td>
      <td>1803.60</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Analysing the data on Cpu Vendors

plt.title("Market Share of each CPU vendor", fontsize=20)
df['Cpu Vendor'].value_counts().plot(kind = 'bar', rot = 0, sort_columns = True, figsize = (15,10), fontsize = 12)
df['Cpu Vendor'].value_counts()
```




    Intel      1240
    AMD          62
    Samsung       1
    Name: Cpu Vendor, dtype: int64




    
![png](output_43_1.png)
    



```python
# Same as that of before but as a percentage pie-chart

plt.title("Market Share of each CPU vendor", fontsize=20)
df['Cpu Vendor'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize = (15,10), fontsize = 12).set_ylabel('')
df['Cpu Vendor'].value_counts(normalize = True)*100
```




    Intel      95.17
    AMD         4.76
    Samsung     0.08
    Name: Cpu Vendor, dtype: float64




    
![png](output_44_1.png)
    


We can see that from the above data that Intel chipset dominate the market by about 95.2% with AMD following up with around 4.8%. We can see Samsung is the least with only 0.08%(i.e 1 laptop). 

We can also see that the chipset market has only three players in total, so not much options to choose from.


```python
# Analysing the data on Cpu Types

plt.title("Number of laptops with each CPU Type", fontsize=20)
df['Cpu Type'].value_counts().plot(kind = 'bar', rot = 45, sort_columns = True, figsize = (15,10), fontsize = 12)
df['Cpu Type'].value_counts()
```




    Core          1105
    Celeron         88
    Pentium         30
    A9-Series       17
    Atom            13
    A6-Series       11
    E-Series         9
    A12-Series       8
    A10-Series       6
    A8-Series        4
    Xeon             4
    Ryzen            4
    FX               2
    Cortex           1
    A4-Series        1
    Name: Cpu Type, dtype: int64




    
![png](output_46_1.png)
    



```python
# Same as that of before but as a percentage pie-chart

plt.title("Number of laptops with each CPU Type", fontsize=20)
df['Cpu Type'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize = (15,10), fontsize = 12).set_ylabel('')
df['Cpu Type'].value_counts(normalize = True)*100
```




    Core          84.80
    Celeron        6.75
    Pentium        2.30
    A9-Series      1.30
    Atom           1.00
    A6-Series      0.84
    E-Series       0.69
    A12-Series     0.61
    A10-Series     0.46
    A8-Series      0.31
    Xeon           0.31
    Ryzen          0.31
    FX             0.15
    Cortex         0.08
    A4-Series      0.08
    Name: Cpu Type, dtype: float64




    
![png](output_47_1.png)
    


We can see that the Core type has the most share and it is dominating the others by a vast majority.


```python
# Analysing the data on Cpu Series

plt.title("Market Share of each CPU series", fontsize=20)
df['Cpu Series'].value_counts().plot(kind = 'bar', rot = 60, sort_columns = True, figsize = (15,10), fontsize = 12)
df['Cpu Series'].value_counts()
```




    i7           527
    i5           423
    i3           136
    Dual Core     83
    Quad Core     35
    M             19
    9420          12
    x5            10
    9220           8
    9720P          7
    E2             4
    7410           4
    9410           3
    1700           3
    X5             2
    A10            2
    7110           2
    9600P          2
    A6             2
    9620P          2
    A9             2
    E3-1505M       2
    E3-1535M       2
    Z8350          1
    9000e          1
    9700P          1
    9830P          1
    6110           1
    8800P          1
    7210           1
    A72&A53        1
    1600           1
    7310           1
    9000           1
    Name: Cpu Series, dtype: int64




    
![png](output_49_1.png)
    


We can see that i7 holds the most market share followed by i5, i3, dual core, quad core and so on which seems obvious given that the majority share on the Cpu market is held by Intel.


```python
# Performing analysis on the Laptop Cpu Model

print(f"There are {len(df[df['Cpu Model'] == 'None'])} laptops whose Cpu Model is not available or is unspecified.")
print("So let's perform our analysis on those laptops that have their Cpu Model properly mentioned")
Screen_withoutNone = df[df['Cpu Model'] != 'None']
plt.title("Cpu Models Available in Laptops", fontsize=20)
Screen_withoutNone['Cpu Model'].value_counts().plot(kind = 'bar', rot = 90, sort_columns = True, figsize = (15,10), fontsize = 12)
Screen_withoutNone['Cpu Model'].value_counts()
```

    There are 70 laptops whose Cpu Model is not available or is unspecified.
    So let's perform our analysis on those laptops that have their Cpu Model properly mentioned
    




    7200U     193
    7700HQ    147
    7500U     136
    6006U      81
    8550U      73
             ... 
    6260U       1
    5           1
    6Y54        1
    4405U       1
    4405Y       1
    Name: Cpu Model, Length: 68, dtype: int64




    
![png](output_51_2.png)
    


We can see that 7200U model is the most liked, followed by 7700HQ and 7500U with 6330HQ being the least preferred


```python
# Analyzing the data on CPU Speed

# Displaying the unique Cpu speed values

df['Cpu Speed (GHz)'].unique()
```




    array([2.3 , 1.8 , 2.5 , 2.7 , 3.1 , 3.  , 2.2 , 1.6 , 2.  , 2.8 , 1.2 ,
           2.9 , 2.4 , 1.44, 1.5 , 1.9 , 1.1 , 1.3 , 2.6 , 3.6 , 3.2 , 1.  ,
           2.1 , 0.9 , 1.92])




```python
# Displaying the maximum and minimum Cpu speed values

print('Min Cpu Speed available :- ',df['Cpu Speed (GHz)'].min(),'GHz')
print('Max Cpu Speed available :- ',df['Cpu Speed (GHz)'].max(),'GHz')
```

    Min Cpu Speed available :-  0.9 GHz
    Max Cpu Speed available :-  3.6 GHz
    


```python
# Drawing a histogram to show the distribution of Cpu speeds

Cpu_speed = [0]*8
for i in df['Cpu Speed (GHz)']:
    Cpu_speed[int(i//0.5)] += 1
fig, ax = plt.subplots(figsize =(15,10))
ax.set_title('Count of Laptops grouped based on their Cpu Speed', fontsize=20)
ax.set_xlabel('Cpu Speed', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.hist(df['Cpu Speed (GHz)'], bins = [0.5*i for i in range(9)])
plt.show()
print(Cpu_speed)
```


    
![png](output_55_0.png)
    


    [0, 4, 87, 225, 238, 721, 23, 5]
    

We can see that the majority of the Cpu processers have a Cpu speed of 2.5 to 3.0, which is enough for an average user, striking a balance between cost and effeciency. 

Almost all the laptops have a processor with a Cpu speed of between 1.0 to 3.0

#### Now let us look at the RAM of the laptops


```python
# Here we see all the different RAM's available

df['Ram'].unique()
```




    array(['8GB', '16GB', '4GB', '2GB', '12GB', '6GB', '32GB', '24GB', '64GB'],
          dtype=object)




```python
# Here we are looking at the market share of each Ram

plt.title("Market Share of each Laptop based on RAM", fontsize=20)
df['Ram'].value_counts().plot(kind = 'bar', rot = 0, sort_columns = True, figsize = (15,10), fontsize = 12)
df['Ram'].value_counts()
```




    8GB     619
    4GB     375
    16GB    200
    6GB      41
    12GB     25
    2GB      22
    32GB     17
    24GB      3
    64GB      1
    Name: Ram, dtype: int64




    
![png](output_59_1.png)
    



```python
# Same as that of before but as a percentage pie-chart

plt.title("Market Share of each Laptop based on RAM", fontsize=20)
df['Ram'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize = (15,12), fontsize = 12).set_ylabel('')
df['Ram'].value_counts(normalize = True)*100
```




    8GB     47.51
    4GB     28.78
    16GB    15.35
    6GB      3.15
    12GB     1.92
    2GB      1.69
    32GB     1.30
    24GB     0.23
    64GB     0.08
    Name: Ram, dtype: float64




    
![png](output_60_1.png)
    


We can see that most of the users perfer 8GB laptops, since they strike an balance between the price and effeciency, followed by 4GB and 16GB ones.


```python
# The value of the RAM is a string but the having it as a numerical data makes more sense and so we convert the data to a numerical one here

df['Ram'] = df['Ram'].replace('[GB]', '', regex=True)
df['Ram'] = pd.to_numeric(df['Ram'])
df.rename(columns={'Ram':'Ram (GB)'}, inplace=True)
```

#### Now let us look at the Memory of the laptops

This is another interesting column, if we observe closely we have two memory values for a few columns, this signifies that the laptop has an additional memory, also known as the secondary memory and we also have the type of memory for each memory device.

So we segregate everything into 4 different columns namely Primary Memory, Primary Memory Type, Secondary Memory and Secondary Memory Type and fill the missing values with None.


```python
# Here we are splitting the column of Memory into their constituent types

# First we are splitting the memory into Primary Memory and Seconday Memory
# We then convert Primary Memory into Primary Memory and Primary Memory Type, and do the same for Secondary Memory too

Memory = df['Memory'].str.split('+', expand = True)
for i in range(len(Memory[1])):
    if Memory[1][i] != None:
        Memory[1][i] = Memory[1][i].strip()
df.insert(16, 'Secondary Memory', Memory[1])

    
df['Memory'] = Memory[0]
df.rename(columns = {'Memory': 'Primary Memory'}, inplace = True)

Sec_Memory = df['Secondary Memory'].str.split(' ',1, expand = True)
df.insert(17, 'Secondary Memory Type', Sec_Memory[1])
df['Secondary Memory'] = Sec_Memory[0]

Pri_Memory = df['Primary Memory'].str.split(' ',1, expand = True)
df.insert(16, 'Primary Memory Type', Pri_Memory[1])
df['Primary Memory'] = Pri_Memory[0]

# A few values in Primary Memory and SEcondary Memory have TB in them, where 1 TB means 1024 GB, so we convert them all into GB for uniformity and 
# then we convert the columns of Primary Memory and Secondary Memory into numerical values

df['Primary Memory'] = df['Primary Memory'].replace('1TB', '1024GB')
df['Primary Memory'] = df['Primary Memory'].replace('1.0TB', '1024GB')
df['Primary Memory'] = df['Primary Memory'].replace('2TB', '2048GB')
df['Primary Memory'] = df['Primary Memory'].replace('[GB]', '', regex=True)
df['Primary Memory'] = pd.to_numeric(df['Primary Memory'], downcast = 'integer')
df['Primary Memory'] = df['Primary Memory'].fillna('None')
df.rename(columns={'Primary Memory':'Primary Memory (GB)'}, inplace=True)

df['Secondary Memory'] = df['Secondary Memory'].replace('1TB', '1024GB')
df['Secondary Memory'] = df['Secondary Memory'].replace('1.0TB', '1024GB')
df['Secondary Memory'] = df['Secondary Memory'].replace('2TB', '2048GB')
df['Secondary Memory'] = df['Secondary Memory'].replace('[GB]', '', regex=True)
df['Secondary Memory'] = pd.to_numeric(df['Secondary Memory'], downcast = 'integer')
df['Secondary Memory'] = df['Secondary Memory'].fillna('None')
df.rename(columns={'Secondary Memory':'Secondary Memory (GB)'}, inplace=True)
```


```python
# Displaying the dataframe with all the analysis done as of now

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>laptop_ID</th>
      <th>Company</th>
      <th>Product</th>
      <th>TypeName</th>
      <th>Inches</th>
      <th>Screen Width</th>
      <th>Screen Height</th>
      <th>Screen Type</th>
      <th>TouchScreen</th>
      <th>Cpu Vendor</th>
      <th>Cpu Type</th>
      <th>Cpu Series</th>
      <th>Cpu Model</th>
      <th>Cpu Speed (GHz)</th>
      <th>Ram (GB)</th>
      <th>Primary Memory (GB)</th>
      <th>Primary Memory Type</th>
      <th>Secondary Memory (GB)</th>
      <th>Secondary Memory Type</th>
      <th>Gpu</th>
      <th>OpSys</th>
      <th>Weight</th>
      <th>Price_euros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>2.3</td>
      <td>8</td>
      <td>128</td>
      <td>SSD</td>
      <td>None</td>
      <td>None</td>
      <td>Intel Iris Plus Graphics 640</td>
      <td>macOS</td>
      <td>1.37kg</td>
      <td>1339.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Apple</td>
      <td>Macbook Air</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>1440</td>
      <td>900</td>
      <td>None</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>1.8</td>
      <td>8</td>
      <td>128</td>
      <td>Flash Storage</td>
      <td>None</td>
      <td>None</td>
      <td>Intel HD Graphics 6000</td>
      <td>macOS</td>
      <td>1.34kg</td>
      <td>898.94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>HP</td>
      <td>250 G6</td>
      <td>Notebook</td>
      <td>15.6</td>
      <td>1920</td>
      <td>1080</td>
      <td>Full HD</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>7200U</td>
      <td>2.5</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>None</td>
      <td>None</td>
      <td>Intel HD Graphics 620</td>
      <td>No OS</td>
      <td>1.86kg</td>
      <td>575.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>15.4</td>
      <td>2880</td>
      <td>1800</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i7</td>
      <td>None</td>
      <td>2.7</td>
      <td>16</td>
      <td>512</td>
      <td>SSD</td>
      <td>None</td>
      <td>None</td>
      <td>AMD Radeon Pro 455</td>
      <td>macOS</td>
      <td>1.83kg</td>
      <td>2537.45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>3.1</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>None</td>
      <td>None</td>
      <td>Intel Iris Plus Graphics 650</td>
      <td>macOS</td>
      <td>1.37kg</td>
      <td>1803.60</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Analysing the data on memory

# Analysing the data on Primary Memory

plt.title("Market Share of each Primary Memory (GB)", fontsize=20)
df['Primary Memory (GB)'].value_counts().plot(kind = 'bar', rot = 0, sort_columns = True, figsize = (15,10), fontsize = 12)
df['Primary Memory (GB)'].value_counts()
```




    256     508
    1024    250
    128     177
    512     140
    500     132
    32       45
    64       17
    2048     16
    16       10
    180       5
    8         1
    240       1
    508       1
    Name: Primary Memory (GB), dtype: int64




    
![png](output_67_1.png)
    



```python
# Analysing the data on Primary Memory Types

plt.title("Market Share of each Primary Memory Type", fontsize=20)
df['Primary Memory Type'].value_counts().plot(kind = 'bar', rot = 0, sort_columns = True, figsize = (15,10), fontsize = 12)
df['Primary Memory Type'].value_counts()
```




    SSD               637
    HDD               374
    SSD               206
    Flash Storage      74
    Hybrid             10
    Flash Storage       1
    HDD                 1
    Name: Primary Memory Type, dtype: int64




    
![png](output_68_1.png)
    



```python
# Analysing the data on Secondary Memory

plt.title("Market Share of each Secondary Memory (GB)", fontsize=20)
Secondary_memory_withoutNone = df['Secondary Memory (GB)'].replace(to_replace='None', value=np.nan).dropna()
Secondary_memory_withoutNone.value_counts().plot(kind = 'bar', rot = 0, sort_columns = True, figsize = (15,10), fontsize = 12)
Secondary_memory_withoutNone.value_counts()
```




    1024.0    187
    2048.0     15
    256.0       3
    500.0       2
    512.0       1
    Name: Secondary Memory (GB), dtype: int64




    
![png](output_69_1.png)
    



```python
# Analysing the data on Secondary Memory Types

plt.title("Market Share of each Secondary Memory Type", fontsize=20)
df['Secondary Memory Type'].value_counts().plot(kind = 'bar', rot = 0, sort_columns = True, figsize = (15,10), fontsize = 12)
df['Secondary Memory Type'].value_counts()
```




    HDD       202
    SSD         4
    Hybrid      2
    Name: Secondary Memory Type, dtype: int64




    
![png](output_70_1.png)
    


From the above bar plots we can see that 256.0 GB of SSD primary memory is the most preferred followed by 1024 GB HDD, now coming to the secondary memory we can see that not many laptops have secondary memory and in the ones that have secondary memory we can see many vendors providing 1024 GB of HDD as their secondary memory.

#### Now let us look at the GPU of the laptops


```python
# Segregating the GPU's into Gpu Vendor(i.e Gpu Manufacturer) and Gpu Model

Gpu_vendor = df['Gpu'].str.split(' ', 1, expand=True)
df.insert(19, 'Gpu Vendor', Gpu_vendor[0])
df['Gpu'] = Gpu_vendor[1]
df.rename(columns = {'Gpu': 'Gpu Model'}, inplace = True)
```


```python
# Here we are looking at the market share of each Gpu vendor

plt.title("Market Share of each GPU Manufacturer", fontsize=20)
df['Gpu Vendor'].value_counts().plot(kind = 'bar', rot = 0, sort_columns = True, figsize = (15,10), fontsize = 12)
df['Gpu Vendor'].value_counts()
```




    Intel     722
    Nvidia    400
    AMD       180
    ARM         1
    Name: Gpu Vendor, dtype: int64




    
![png](output_74_1.png)
    



```python
# Same as that of before but as a percentage pie-chart

plt.title("Market Share of each GPU Manufacturer", fontsize=20)
df['Gpu Vendor'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize = (15,12), fontsize = 12).set_ylabel('')
df['Gpu Vendor'].value_counts(normalize = True)*100
```




    Intel     55.41
    Nvidia    30.70
    AMD       13.81
    ARM        0.08
    Name: Gpu Vendor, dtype: float64




    
![png](output_75_1.png)
    


Even in the GPU section we can see the market being dominated by Intel followed by Nvidia, the lowest being ARM.


```python
# Looking at all the different GPU Models

df['Gpu Model'].unique()
```




    array(['Iris Plus Graphics 640', 'HD Graphics 6000', 'HD Graphics 620',
           'Radeon Pro 455', 'Iris Plus Graphics 650', 'Radeon R5',
           'Iris Pro Graphics', 'GeForce MX150', 'UHD Graphics 620',
           'HD Graphics 520', 'Radeon Pro 555', 'Radeon R5 M430',
           'HD Graphics 615', 'Radeon Pro 560', 'GeForce 940MX',
           'HD Graphics 400', 'GeForce GTX 1050', 'Radeon R2', 'Radeon 530',
           'GeForce 930MX', 'HD Graphics', 'HD Graphics 500',
           'GeForce 930MX ', 'GeForce GTX 1060', 'GeForce 150MX',
           'Iris Graphics 540', 'Radeon RX 580', 'GeForce 920MX',
           'Radeon R4 Graphics', 'Radeon 520', 'GeForce GTX 1070',
           'GeForce GTX 1050 Ti', 'GeForce MX130', 'R4 Graphics',
           'GeForce GTX 940MX', 'Radeon RX 560', 'GeForce 920M',
           'Radeon R7 M445', 'Radeon RX 550', 'GeForce GTX 1050M',
           'HD Graphics 515', 'Radeon R5 M420', 'HD Graphics 505',
           'GTX 980 SLI', 'R17M-M1-70', 'GeForce GTX 1080', 'Quadro M1200',
           'GeForce 920MX ', 'GeForce GTX 950M', 'FirePro W4190M ',
           'GeForce GTX 980M', 'Iris Graphics 550', 'GeForce 930M',
           'HD Graphics 630', 'Radeon R5 430', 'GeForce GTX 940M',
           'HD Graphics 510', 'HD Graphics 405', 'Radeon RX 540',
           'GeForce GT 940MX', 'FirePro W5130M', 'Quadro M2200M', 'Radeon R4',
           'Quadro M620', 'Radeon R7 M460', 'HD Graphics 530',
           'GeForce GTX 965M', 'GeForce GTX1080', 'GeForce GTX1050 Ti',
           'GeForce GTX 960M', 'Radeon R2 Graphics', 'Quadro M620M',
           'GeForce GTX 970M', 'GeForce GTX 960<U+039C>', 'Graphics 620',
           'GeForce GTX 960', 'Radeon R5 520', 'Radeon R7 M440', 'Radeon R7',
           'Quadro M520M', 'Quadro M2200', 'Quadro M2000M', 'HD Graphics 540',
           'Quadro M1000M', 'Radeon 540', 'GeForce GTX 1070M',
           'GeForce GTX1060', 'HD Graphics 5300', 'Radeon R5 M420X',
           'Radeon R7 Graphics', 'GeForce 920', 'GeForce 940M',
           'GeForce GTX 930MX', 'Radeon R7 M465', 'Radeon R3',
           'GeForce GTX 1050Ti', 'Radeon R7 M365X', 'Radeon R9 M385',
           'HD Graphics 620 ', 'Quadro 3000M', 'GeForce GTX 980 ',
           'Radeon R5 M330', 'FirePro W4190M', 'FirePro W6150M',
           'Radeon R5 M315', 'Quadro M500M', 'Radeon R7 M360',
           'Quadro M3000M', 'GeForce 960M', 'Mali T860 MP4'], dtype=object)



Drawing a bar plot and checking the share of each model is not feasible given the number of unique models available, splitting the model into its components is also not feasible since the model names are not following a proper pattern and so we are avoiding it.

#### Now let us look at the Operating System of the laptops


```python
# Looking at all the different Operating Systems

df['OpSys'].unique()
```




    array(['macOS', 'No OS', 'Windows 10', 'Mac OS X', 'Linux', 'Android',
           'Windows 10 S', 'Chrome OS', 'Windows 7'], dtype=object)




```python
# Here we are looking at the share of each OS

plt.title("Market Share of each Laptop based on their OS", fontsize=20)
df['OpSys'].value_counts().plot(kind = 'bar', rot = 0, sort_columns = True, figsize = (15,10), fontsize = 12)
df['OpSys'].value_counts()
```




    Windows 10      1072
    No OS             66
    Linux             62
    Windows 7         45
    Chrome OS         27
    macOS             13
    Windows 10 S       8
    Mac OS X           8
    Android            2
    Name: OpSys, dtype: int64




    
![png](output_81_1.png)
    



```python
# Same as that of before but as a percentage pie-chart

plt.title("Market Share of each Laptop based on their OS", fontsize=20)
df['OpSys'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize = (15,12), fontsize = 12).set_ylabel('')
df['OpSys'].value_counts(normalize = True)*100
```




    Windows 10      82.27
    No OS            5.07
    Linux            4.76
    Windows 7        3.45
    Chrome OS        2.07
    macOS            1.00
    Windows 10 S     0.61
    Mac OS X         0.61
    Android          0.15
    Name: OpSys, dtype: float64




    
![png](output_82_1.png)
    


From the above graphs we can see that Windows 10 is the OS provided by most of the vendors. It has a whopping 82.3% market share while all the others are having a minority share of 5.1%, 4.8% and so on.

#### Now let us look at the Weight of the laptops


```python
# Here we are converting the column weight with consists of strings into numeric values

df['Weight'] = df['Weight'].replace('[kg]', '', regex=True)
df['Weight'] = pd.to_numeric(df['Weight'])
df.rename(columns={'Weight':'Weight (Kg)'}, inplace=True)
```


```python
# Checking the minimum and maximum weights of laptops

print('Min weight available :- ',df['Weight (Kg)'].min(),'Kg')
print('Max weight available :- ',df['Weight (Kg)'].max(),'Kg')
```

    Min weight available :-  0.69 Kg
    Max weight available :-  4.7 Kg
    

So now we divide the laptops into seperate categories based on their weights, for this we take a range of 0.5 kgs. So if we start from 0.5 Kgs and have to go till 4.7 Kgs in intervals of 0.5 Kgs each thereby we would end up with 9 different categories.


```python
# Here we are grouping the weights into factors of 0.5 and plotting the number of laptops in each catergories

Weights_count = [0]*10
for i in df['Weight (Kg)']:
    Weights_count[int(i//0.5)] += 1
fig, ax = plt.subplots(figsize =(15,10))
ax.set_title('Count of Laptops grouped based on their weights', fontsize=20)
ax.set_xlabel('Weight', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.hist(df['Weight (Kg)'], bins = [0.5*i for i in range(11)])
plt.show()
print(Weights_count)
```


    
![png](output_88_0.png)
    


    [0, 18, 296, 272, 473, 155, 43, 11, 29, 6]
    

From the above histogram we can see that majority of the laptops have their weights between 1 and 3 with most falling in the range of 2 to 2.5 kgs. This makes sense given the fact that laptops being popularly known as portable computers are not so easily portable if they are heavy and 1 to 2 or 2.5 kgs is usually the sweet spot, where cost and portability strike a balance.

#### Visualizing the dataset using Heat-Map


```python
# Displaying a heat-map to check the relationship between numerical features.

plt.figure()
plt.figure(figsize = (15,10))
plt.title('Correlation among columns',fontsize = 20)
laptop = sns.heatmap(df.corr(), cmap='coolwarm', annot=True, linewidths=.1)
laptop.set_xticklabels(laptop.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
```


    <Figure size 432x288 with 0 Axes>



    
![png](output_91_1.png)
    


From the heat-map we can see that the features that have a huge meaningful influence on the price of the laptop are the Screen Width and Screen Height, Cpu speed, Ram and Weight. This heat-map shows the influence of only the numerical values whereas a lot of other features such as Company, the CPU and GPU vendors, etc,., which also tends to play a role in the price (we will do this analysis in the next section), are still categorical and have to be converted into numerical, which will be performed later on, for training our model.

### Let us see some additional analysis that can be performed on the dataset

#### Let's first look at the Average weight of each Laptop Type


```python
# Drawing a bar graph showing us the Average Weight to each Laptop Type

sns.set(rc={'figure.figsize':(15,10)}, font_scale=1.3)
plt.xticks(rotation=45)
sns.barplot(x=df['TypeName'], y=df['Weight (Kg)']).set_title('Average Weight of each Laptop Type', fontsize = 25)
```




    Text(0.5, 1.0, 'Average Weight of each Laptop Type')




    
![png](output_95_1.png)
    


We can see that Gaming Laptops tend to be a bit heavier then the rest, which makes sense since they come with multiple fans and a host of other components to ensure proper cooling and high performance which is required while gaming. Second heaviest laptops are that of Workstation, followed by Notebooks and lightest are Netbooks.

#### Now let's look at the Average price of each Laptop Type


```python
# Drawing a bar graph showing us the Average Price to each Laptop Type

plt.xticks(rotation=45)
sns.barplot(x=df['TypeName'], y=df['Price_euros']).set_title('Average Price of each Laptop Type', fontsize = 25)
```




    Text(0.5, 1.0, 'Average Price of each Laptop Type')




    
![png](output_98_1.png)
    


From the above bar plot we can see that Workstation Laptops tend to be on the costly side, which makes sense cause Workstation laptops, which are used for heavy end workload, are expected to have huge performance with many features and components which is the reason for the increased cost. They are followed by Gaming Laptops which are also expected to have high performance and features for gaming purposes. The lowest cost is that of the Netbook Laptops, these are light performance laptops whose purpose if just to perform small end tasks like document editing, Internet surfing and other not so heavily computation required tasks.

#### Now let us look at the Average Price of each Laptop Manufacturer


```python
# Drawing a bar graph showing us the Average Price to each Laptop Manufacturer

plt.xticks(rotation=45)
sns.barplot(x=df['Company'], y=df['Price_euros']).set_title('Average Price of each Laptop Product on the basis of its Manufacturer', fontsize = 25)
df.groupby('Company')['Price_euros'].mean()
```




    Company
    Acer          626.78
    Apple        1564.20
    Asus         1104.17
    Chuwi         314.30
    Dell         1186.07
    Fujitsu       729.00
    Google       1677.67
    HP           1067.77
    Huawei       1424.00
    LG           2099.00
    Lenovo       1086.38
    MSI          1728.91
    Mediacom      295.00
    Microsoft    1612.31
    Razer        3346.14
    Samsung      1413.44
    Toshiba      1267.81
    Vero          217.43
    Xiaomi       1133.46
    Name: Price_euros, dtype: float64




    
![png](output_101_1.png)
    


We can see that Razer produced few of the costliest laptops, targeting people who need heavy performance laptops and are ready to compromise with the cost. They are followed by LG, which produced durable laptops with gorgeous displays and MSI, which produces  top end gaming laptops. The lowest being Vero, which produces laptops that carter to the basic requirements, like surfing and document editing,etc,., for the people who do not want to spend much on laptops.

#### Now let us look at the Average Price of each laptop based on it's GPU provider


```python
plt.xticks(rotation=45)
sns.barplot(x=df['Gpu Vendor'], y=df['Price_euros']).set_title('Average Price of each Laptop Product on the basis of GPU manufacturer', fontsize = 25)
df.groupby('Gpu Vendor')['Price_euros'].mean()
```




    Gpu Vendor
    AMD        775.65
    ARM        659.00
    Intel     1008.23
    Nvidia    1489.87
    Name: Price_euros, dtype: float64




    
![png](output_104_1.png)
    


We can see that Nvidia's GPU's are the costliest and the reason is because they produce heavy GPU's suited for various tasks ranging from hardcore gaming to lightweight editing and are quite popular for their speed and efficiency. Nvidia is considered the leader at making GPU's. They are followed by Intel who are the leaders at making CPU chip-sets. We can also notice that that entire GPU market has only 4 players which also makes sense since GPU's too require a heavy capital investment and huge running capital to maintain the infrastructure, which acts as a huge barrier thereby preventing new players from entering the market. 

#### Now let us look at the Average Price of each laptop based on it's Screen Type


```python
Screen_withoutNone = df[df['Screen Type'] != 'None']
plt.xticks(rotation=75)
sns.barplot(x=Screen_withoutNone['Screen Type'], y=df['Price_euros']).set_title('Average Price of each Laptop Product on the basis of Screen Type', fontsize = 25)
Screen_withoutNone.groupby('Screen Type')['Price_euros'].mean()
```




    Screen Type
    4K Ultra HD                          2629.93
    Full HD                              1175.46
    IPS Panel                            1369.64
    IPS Panel 4K Ultra HD                2329.41
    IPS Panel Full HD                    1323.98
    IPS Panel Quad HD+                   1579.79
    IPS Panel Retina Display             1657.85
    IPS Panel Touchscreen                1107.84
    IPS Panel Touchscreen 4K Ultra HD    1674.84
    Quad HD+                             1626.06
    Name: Price_euros, dtype: float64




    
![png](output_107_1.png)
    


We can see that laptops with 4K Ultra HD cost a lot, followed by IPS Panel 4K Ultra HD, the cheapest being the ones with IPS Panel Touchscreen. For all normal purposes Full HD screen or an IPS full HD screen is sufficient.

#### Now let us look at the Average Price of each laptop based on their Operating System


```python
plt.xticks(rotation=45)
sns.barplot(x=df['OpSys'], y=df['Price_euros']).set_title('Average Price of each Laptop Product on the basis of their Operating System', fontsize = 25)
df.groupby('OpSys')['Price_euros'].mean()
```




    OpSys
    Android          434.00
    Chrome OS        553.59
    Linux            617.07
    Mac OS X        1262.87
    No OS            587.97
    Windows 10      1168.14
    Windows 10 S    1286.48
    Windows 7       1686.65
    macOS           1749.63
    Name: Price_euros, dtype: float64




    
![png](output_110_1.png)
    


We can see MacOS has more asking price than the rest of the Operating System, this makes sense since MacOS is the proprietary software of Apple and is only available in their laptops(Macbooks), and Macbooks tend to be a bit on the costly side unlike Windows 10 which is available in almost all the laptops being costly or cheap. The lowest being Android, since many do not prefer Android in laptops.

#### Now let us look at the Average Price of each laptop based on it's Memory


```python
plt.title("Average Price of each Laptop based on their Primary and Secondary Memory", fontsize=20)
Memory_price = df.groupby(['Primary Memory (GB)','Primary Memory Type','Secondary Memory (GB)','Secondary Memory Type'])['Price_euros'].mean()
Memory_price.plot(kind = 'bar', figsize = (15,10), rot = 75, fontsize = 12)
print(Memory_price)
plt.show()
```

    Primary Memory (GB)  Primary Memory Type  Secondary Memory (GB)  Secondary Memory Type
    64                   Flash Storage        1024.0                 HDD                      1993.00
    128                  SSD                  1024.0                 HDD                      1266.82
                                              2048.0                 HDD                       977.95
    256                  SSD                  256.0                  SSD                      1288.50
                                              500.0                  HDD                      1497.17
                                              1024.0                 HDD                      1856.15
                                                                     Hybrid                   2749.99
                                              2048.0                 HDD                      1535.91
    512                  SSD                  256.0                  SSD                      1607.96
                                              512.0                  SSD                      1499.00
                                              1024.0                 HDD                      2500.45
                                                                     Hybrid                   3240.00
                                              2048.0                 HDD                      1849.00
    1024                 HDD                  1024.0                 HDD                       621.45
                         SSD                  1024.0                 HDD                      3624.10
    Name: Price_euros, dtype: float64
    


    
![png](output_113_1.png)
    


From the bar plot we can see that 1024 GB SSD as primary memory and 1024 GB HDD as the secondary memory have the highest price while the lowest being 1024 GB HDD as primary memory and 1024 GB HDD as the secondary memory, the reason for which lies on the cost of SDD's. SDD's are very fast and effective at data writing and retrieval then the traditional HDD, and due to this fact they are very expensive than the traditional HDD. It is to strike a balance between the performance gains and the cost that people usually prefer 128 GB SDD as their primary memory and 2048 GB HDD as their secondary memory, which is also evident from the above bar plot, being the next cheapest option after 1024 GB HDD primary memory and 1024 GB HDD secondary memory.

### Saving our preprocessed dataset as a .csv file


```python
# Now that we have done the required preprocessing and some data analysis(where we were able to arrive at some useful conclusions) we now
# proceed to saving the preprocessed dataset to use in training our models, we save our model for the sole reason being that we would not
# have to go through the entire process above again while training and testing our model.

df.to_csv('Laptop_data_Preprocessed.csv', index = False)
```

## Building regression models and testing them


```python
# Reading the dataset containing the preprocessed data obtained from above

df = pd.read_csv('Laptop_data_Preprocessed.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>laptop_ID</th>
      <th>Company</th>
      <th>Product</th>
      <th>TypeName</th>
      <th>Inches</th>
      <th>Screen Width</th>
      <th>Screen Height</th>
      <th>Screen Type</th>
      <th>TouchScreen</th>
      <th>Cpu Vendor</th>
      <th>Cpu Type</th>
      <th>Cpu Series</th>
      <th>Cpu Model</th>
      <th>Cpu Speed (GHz)</th>
      <th>Ram (GB)</th>
      <th>Primary Memory (GB)</th>
      <th>Primary Memory Type</th>
      <th>Secondary Memory (GB)</th>
      <th>Secondary Memory Type</th>
      <th>Gpu Vendor</th>
      <th>Gpu Model</th>
      <th>OpSys</th>
      <th>Weight (Kg)</th>
      <th>Price_euros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>2.3</td>
      <td>8</td>
      <td>128</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>Intel</td>
      <td>Iris Plus Graphics 640</td>
      <td>macOS</td>
      <td>1.37</td>
      <td>1339.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Apple</td>
      <td>Macbook Air</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>1440</td>
      <td>900</td>
      <td>None</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>1.8</td>
      <td>8</td>
      <td>128</td>
      <td>Flash Storage</td>
      <td>None</td>
      <td>NaN</td>
      <td>Intel</td>
      <td>HD Graphics 6000</td>
      <td>macOS</td>
      <td>1.34</td>
      <td>898.94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>HP</td>
      <td>250 G6</td>
      <td>Notebook</td>
      <td>15.6</td>
      <td>1920</td>
      <td>1080</td>
      <td>Full HD</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>7200U</td>
      <td>2.5</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>Intel</td>
      <td>HD Graphics 620</td>
      <td>No OS</td>
      <td>1.86</td>
      <td>575.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>15.4</td>
      <td>2880</td>
      <td>1800</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i7</td>
      <td>None</td>
      <td>2.7</td>
      <td>16</td>
      <td>512</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>AMD</td>
      <td>Radeon Pro 455</td>
      <td>macOS</td>
      <td>1.83</td>
      <td>2537.45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>3.1</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>Intel</td>
      <td>Iris Plus Graphics 650</td>
      <td>macOS</td>
      <td>1.37</td>
      <td>1803.60</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1298</th>
      <td>1316</td>
      <td>Lenovo</td>
      <td>Yoga 500-14ISK</td>
      <td>2 in 1 Convertible</td>
      <td>14.0</td>
      <td>1920</td>
      <td>1080</td>
      <td>IPS Panel Full HD</td>
      <td>Yes</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i7</td>
      <td>6500U</td>
      <td>2.5</td>
      <td>4</td>
      <td>128</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>Intel</td>
      <td>HD Graphics 520</td>
      <td>Windows 10</td>
      <td>1.80</td>
      <td>638.00</td>
    </tr>
    <tr>
      <th>1299</th>
      <td>1317</td>
      <td>Lenovo</td>
      <td>Yoga 900-13ISK</td>
      <td>2 in 1 Convertible</td>
      <td>13.3</td>
      <td>3200</td>
      <td>1800</td>
      <td>IPS Panel Quad HD+</td>
      <td>Yes</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i7</td>
      <td>6500U</td>
      <td>2.5</td>
      <td>16</td>
      <td>512</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>Intel</td>
      <td>HD Graphics 520</td>
      <td>Windows 10</td>
      <td>1.30</td>
      <td>1499.00</td>
    </tr>
    <tr>
      <th>1300</th>
      <td>1318</td>
      <td>Lenovo</td>
      <td>IdeaPad 100S-14IBR</td>
      <td>Notebook</td>
      <td>14.0</td>
      <td>1366</td>
      <td>768</td>
      <td>None</td>
      <td>No</td>
      <td>Intel</td>
      <td>Celeron</td>
      <td>Dual Core</td>
      <td>N3050</td>
      <td>1.6</td>
      <td>2</td>
      <td>64</td>
      <td>Flash Storage</td>
      <td>None</td>
      <td>NaN</td>
      <td>Intel</td>
      <td>HD Graphics</td>
      <td>Windows 10</td>
      <td>1.50</td>
      <td>229.00</td>
    </tr>
    <tr>
      <th>1301</th>
      <td>1319</td>
      <td>HP</td>
      <td>15-AC110nv (i7-6500U/6GB/1TB/Radeon</td>
      <td>Notebook</td>
      <td>15.6</td>
      <td>1366</td>
      <td>768</td>
      <td>None</td>
      <td>No</td>
      <td>Intel</td>
      <td>Core</td>
      <td>i7</td>
      <td>6500U</td>
      <td>2.5</td>
      <td>6</td>
      <td>1024</td>
      <td>HDD</td>
      <td>None</td>
      <td>NaN</td>
      <td>AMD</td>
      <td>Radeon R5 M330</td>
      <td>Windows 10</td>
      <td>2.19</td>
      <td>764.00</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>1320</td>
      <td>Asus</td>
      <td>X553SA-XX031T (N3050/4GB/500GB/W10)</td>
      <td>Notebook</td>
      <td>15.6</td>
      <td>1366</td>
      <td>768</td>
      <td>None</td>
      <td>No</td>
      <td>Intel</td>
      <td>Celeron</td>
      <td>Dual Core</td>
      <td>N3050</td>
      <td>1.6</td>
      <td>4</td>
      <td>500</td>
      <td>HDD</td>
      <td>None</td>
      <td>NaN</td>
      <td>Intel</td>
      <td>HD Graphics</td>
      <td>Windows 10</td>
      <td>2.20</td>
      <td>369.00</td>
    </tr>
  </tbody>
</table>
<p>1303 rows × 24 columns</p>
</div>



### Encoding the data, removing the unwanted features and scaling the data

Many machine learning algorithms cannot operate on label data directly. They require all input variables and output variables to be numeric.

In general, this is mostly a constraint of the efficient implementation of machine learning algorithms rather than hard limitations on the algorithms themselves.

This means that categorical data must be converted to a numerical form. If the categorical variable is an output variable, you may also want to convert predictions by the model back into a categorical form in order to present them or use them in some application and this is done in two ways either through Label Encoding or One-Hot Encoding.

1. Label Encoding
Each unique category value is assigned an integer value.

    For example, “red” is 1, “green” is 2, and “blue” is 3.

    This is called a label encoding or an integer encoding and is easily reversible.

    For some variables, this may be enough for some it creates a problem that is a ML model assumes assumes higher the categorical value, better the category and this sometimes results in a model with less accuracy.

    The integer values have a natural ordered relationship between each other and machine learning algorithms may be able to understand and harness this relationship.

2. One-Hot Encoding
For categorical variables where no such ordinal relationship exists, the integer encoding is not enough.

    In fact, using this encoding and allowing the model to assume a natural ordering between categories may result in poor performance or unexpected results (predictions halfway between categories).

    In this case, a one-hot encoding can be applied to the integer representation. This is where the integer encoded variable is removed and a new binary variable is added for each unique integer value.


```python
# Performing One-Hot Encoding on the column Cpu Vendor and Gpu Vendor

Onehot = OneHotEncoder(sparse = False)
Onehot.fit(df['Cpu Vendor'].unique().reshape(-1,1))
transformed = Onehot.transform(df['Cpu Vendor'].to_numpy().reshape(-1,1))
Vendors = pd.DataFrame(transformed, columns = ['Cpu_Vendor_'+ str(i.replace('x0_', '')) for i in Onehot.get_feature_names()])
Vendors = Vendors.drop(['Cpu_Vendor_Samsung'], axis=1)
df.insert(9, 'Cpu_Vendor_AMD', Vendors['Cpu_Vendor_AMD'])
df.insert(10, 'Cpu_Vendor_Intel', Vendors['Cpu_Vendor_Intel'])
df = df.drop(['Cpu Vendor'], axis = 1)

Onehot.fit(df['Gpu Vendor'].unique().reshape(-1,1))
transformed = Onehot.transform(df['Gpu Vendor'].to_numpy().reshape(-1,1))
Vendors = pd.DataFrame(transformed, columns = ['Gpu_Vendor_'+ str(i.replace('x0_', '')) for i in Onehot.get_feature_names()])
Vendors = Vendors.drop(['Gpu_Vendor_ARM'], axis=1)
df.insert(21, 'Gpu_Vendor_AMD', Vendors['Gpu_Vendor_AMD'])
df.insert(22, 'Gpu_Vendor_Intel', Vendors['Gpu_Vendor_Intel'])
df.insert(23, 'Gpu_Vendor_Nvidia', Vendors['Gpu_Vendor_Nvidia'])
df = df.drop(['Gpu Vendor'], axis = 1)

# Dropping the column laptop_ID since it is just used to identify the product(serves as an index) and has no relevance in the training

df = df.drop(['laptop_ID'], axis = 1)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>Product</th>
      <th>TypeName</th>
      <th>Inches</th>
      <th>Screen Width</th>
      <th>Screen Height</th>
      <th>Screen Type</th>
      <th>TouchScreen</th>
      <th>Cpu_Vendor_AMD</th>
      <th>Cpu_Vendor_Intel</th>
      <th>Cpu Type</th>
      <th>Cpu Series</th>
      <th>Cpu Model</th>
      <th>Cpu Speed (GHz)</th>
      <th>Ram (GB)</th>
      <th>Primary Memory (GB)</th>
      <th>Primary Memory Type</th>
      <th>Secondary Memory (GB)</th>
      <th>Secondary Memory Type</th>
      <th>Gpu_Vendor_AMD</th>
      <th>Gpu_Vendor_Intel</th>
      <th>Gpu_Vendor_Nvidia</th>
      <th>Gpu Model</th>
      <th>OpSys</th>
      <th>Weight (Kg)</th>
      <th>Price_euros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>2.3</td>
      <td>8</td>
      <td>128</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Iris Plus Graphics 640</td>
      <td>macOS</td>
      <td>1.37</td>
      <td>1339.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apple</td>
      <td>Macbook Air</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>1440</td>
      <td>900</td>
      <td>None</td>
      <td>No</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>1.8</td>
      <td>8</td>
      <td>128</td>
      <td>Flash Storage</td>
      <td>None</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>HD Graphics 6000</td>
      <td>macOS</td>
      <td>1.34</td>
      <td>898.94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HP</td>
      <td>250 G6</td>
      <td>Notebook</td>
      <td>15.6</td>
      <td>1920</td>
      <td>1080</td>
      <td>Full HD</td>
      <td>No</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Core</td>
      <td>i5</td>
      <td>7200U</td>
      <td>2.5</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>HD Graphics 620</td>
      <td>No OS</td>
      <td>1.86</td>
      <td>575.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>15.4</td>
      <td>2880</td>
      <td>1800</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Core</td>
      <td>i7</td>
      <td>None</td>
      <td>2.7</td>
      <td>16</td>
      <td>512</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Radeon Pro 455</td>
      <td>macOS</td>
      <td>1.83</td>
      <td>2537.45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Apple</td>
      <td>MacBook Pro</td>
      <td>Ultrabook</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>IPS Panel Retina Display</td>
      <td>No</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Core</td>
      <td>i5</td>
      <td>None</td>
      <td>3.1</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Iris Plus Graphics 650</td>
      <td>macOS</td>
      <td>1.37</td>
      <td>1803.60</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1298</th>
      <td>Lenovo</td>
      <td>Yoga 500-14ISK</td>
      <td>2 in 1 Convertible</td>
      <td>14.0</td>
      <td>1920</td>
      <td>1080</td>
      <td>IPS Panel Full HD</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Core</td>
      <td>i7</td>
      <td>6500U</td>
      <td>2.5</td>
      <td>4</td>
      <td>128</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>HD Graphics 520</td>
      <td>Windows 10</td>
      <td>1.80</td>
      <td>638.00</td>
    </tr>
    <tr>
      <th>1299</th>
      <td>Lenovo</td>
      <td>Yoga 900-13ISK</td>
      <td>2 in 1 Convertible</td>
      <td>13.3</td>
      <td>3200</td>
      <td>1800</td>
      <td>IPS Panel Quad HD+</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Core</td>
      <td>i7</td>
      <td>6500U</td>
      <td>2.5</td>
      <td>16</td>
      <td>512</td>
      <td>SSD</td>
      <td>None</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>HD Graphics 520</td>
      <td>Windows 10</td>
      <td>1.30</td>
      <td>1499.00</td>
    </tr>
    <tr>
      <th>1300</th>
      <td>Lenovo</td>
      <td>IdeaPad 100S-14IBR</td>
      <td>Notebook</td>
      <td>14.0</td>
      <td>1366</td>
      <td>768</td>
      <td>None</td>
      <td>No</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Celeron</td>
      <td>Dual Core</td>
      <td>N3050</td>
      <td>1.6</td>
      <td>2</td>
      <td>64</td>
      <td>Flash Storage</td>
      <td>None</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>HD Graphics</td>
      <td>Windows 10</td>
      <td>1.50</td>
      <td>229.00</td>
    </tr>
    <tr>
      <th>1301</th>
      <td>HP</td>
      <td>15-AC110nv (i7-6500U/6GB/1TB/Radeon</td>
      <td>Notebook</td>
      <td>15.6</td>
      <td>1366</td>
      <td>768</td>
      <td>None</td>
      <td>No</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Core</td>
      <td>i7</td>
      <td>6500U</td>
      <td>2.5</td>
      <td>6</td>
      <td>1024</td>
      <td>HDD</td>
      <td>None</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Radeon R5 M330</td>
      <td>Windows 10</td>
      <td>2.19</td>
      <td>764.00</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>Asus</td>
      <td>X553SA-XX031T (N3050/4GB/500GB/W10)</td>
      <td>Notebook</td>
      <td>15.6</td>
      <td>1366</td>
      <td>768</td>
      <td>None</td>
      <td>No</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Celeron</td>
      <td>Dual Core</td>
      <td>N3050</td>
      <td>1.6</td>
      <td>4</td>
      <td>500</td>
      <td>HDD</td>
      <td>None</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>HD Graphics</td>
      <td>Windows 10</td>
      <td>2.20</td>
      <td>369.00</td>
    </tr>
  </tbody>
</table>
<p>1303 rows × 26 columns</p>
</div>




```python
# Encoding the columns using their Label(Label Encoding) after ordering each column based on the price

col = ['Company', 'Product', 'TypeName', 'Screen Type', 'Cpu Type', 'Cpu Series', 'Gpu Model', 'OpSys']
for feature in col:
    labels_ordered= df.groupby([feature])['Price_euros'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df[feature]=df[feature].map(labels_ordered)
    
# Encoding the value of the column TouchScreen as 1 if it has the value 'Yes' else 0
    
df['TouchScreen'] = df['TouchScreen'].map({'Yes': 1, 'No': 0})

# Replacing the value of the column 'Secondary Memory (GB)' with 0.0 if the laptop has no secondary memory

df['Secondary Memory (GB)'] = df['Secondary Memory (GB)'].replace('None', 0.0)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>Product</th>
      <th>TypeName</th>
      <th>Inches</th>
      <th>Screen Width</th>
      <th>Screen Height</th>
      <th>Screen Type</th>
      <th>TouchScreen</th>
      <th>Cpu_Vendor_AMD</th>
      <th>Cpu_Vendor_Intel</th>
      <th>Cpu Type</th>
      <th>Cpu Series</th>
      <th>Cpu Model</th>
      <th>Cpu Speed (GHz)</th>
      <th>Ram (GB)</th>
      <th>Primary Memory (GB)</th>
      <th>Primary Memory Type</th>
      <th>Secondary Memory (GB)</th>
      <th>Secondary Memory Type</th>
      <th>Gpu_Vendor_AMD</th>
      <th>Gpu_Vendor_Intel</th>
      <th>Gpu_Vendor_Nvidia</th>
      <th>Gpu Model</th>
      <th>OpSys</th>
      <th>Weight (Kg)</th>
      <th>Price_euros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13</td>
      <td>559</td>
      <td>3</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>7</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>12</td>
      <td>27</td>
      <td>None</td>
      <td>2.3</td>
      <td>8</td>
      <td>128</td>
      <td>SSD</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>80</td>
      <td>8</td>
      <td>1.37</td>
      <td>1339.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>333</td>
      <td>3</td>
      <td>13.3</td>
      <td>1440</td>
      <td>900</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>12</td>
      <td>27</td>
      <td>None</td>
      <td>1.8</td>
      <td>8</td>
      <td>128</td>
      <td>Flash Storage</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>45</td>
      <td>8</td>
      <td>1.34</td>
      <td>898.94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>107</td>
      <td>1</td>
      <td>15.6</td>
      <td>1920</td>
      <td>1080</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>12</td>
      <td>27</td>
      <td>7200U</td>
      <td>2.5</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>54</td>
      <td>2</td>
      <td>1.86</td>
      <td>575.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>559</td>
      <td>3</td>
      <td>15.4</td>
      <td>2880</td>
      <td>1800</td>
      <td>7</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>12</td>
      <td>29</td>
      <td>None</td>
      <td>2.7</td>
      <td>16</td>
      <td>512</td>
      <td>SSD</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>95</td>
      <td>8</td>
      <td>1.83</td>
      <td>2537.45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>559</td>
      <td>3</td>
      <td>13.3</td>
      <td>2560</td>
      <td>1600</td>
      <td>7</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>12</td>
      <td>27</td>
      <td>None</td>
      <td>3.1</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>84</td>
      <td>8</td>
      <td>1.37</td>
      <td>1803.60</td>
    </tr>
  </tbody>
</table>
</div>



### Visualizing the dataset using a Heat-map


```python
# Displaying the heatmap to see the relationships of each feature wrt other features

plt.figure()
plt.figure(figsize = (20,20))
plt.title('Correlation among columns after encoding all the labels in each column',fontsize = 20)
laptop = sns.heatmap(df.corr(), annot=True, linewidths=.1)
laptop.set_xticklabels(laptop.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.show()
```


    <Figure size 1080x720 with 0 Axes>



    
![png](output_124_1.png)
    


From the above heat-map we can see that the company, Product, Typename(Laptop Type), Screen Width and Screen Height, Screen Type, Cpu_Vendor, Cpu_Type, Cpu Series, Cpu Speed, Ram, Gpu_Vendor, Gpu Model, Opsys and weight have a good relationship with the Price and can be used to predict price effectively. In the head-map we did before this we were not able to get such a detailed relationship chart since many features where categorical in nature then, but now since we had converted them into numerical with Label Encoding and One-Hot Encoding we are able to see the detailed relationship. 

### Splitting and assigning the data


```python
# Assigning the values to X and y so that we can use them to train our model

X = df.drop(columns = ['Price_euros', 'Cpu Model', 'Primary Memory Type', 'Secondary Memory Type'])
y = df['Price_euros']
```


```python
# Using Standard Scaler to remove the mean and scales each feature to unit variance.

scaler  = StandardScaler()
temp = scaler.fit_transform(X)
X = pd.DataFrame(temp)
```

Standardization is a scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.

This is done so that the variance of the features are in the same range. If a feature’s variance is orders of magnitude more than the variance of other features, that particular feature might dominate other features in the dataset, which is not something we want happening in our model.

The aim here is to to achieve Gaussian with zero mean and unit variance. There are many ways of doing this, two most popular are standardisation and normalisation.


```python
# Spliting out dataset into train and test (80/20 split)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)
```


```python
# Defining the mean absolute percentage error function (a metric used for testing the performance of our model)

def mean_abs_percent_error(x):
    mae = mean_absolute_error(x,y_test)
    mean = y_test.mean()
    percentage_mae = (mae/mean)*100
    return percentage_mae
```

The mean absolute percentage error (MAPE), also known as mean absolute percentage deviation (MAPD), is a measure of prediction accuracy of a forecasting method. 

The closer it is to 0 the better is the prediction capability of our model with 0 being the ideal case.

<br />


The coefficient of determination, denoted R2 or r2 and pronounced "R squared", is the proportion of the variance in the dependent variable that is predictable from the independent variable and is an effective measure to check the accuracy of the model.

The closer the value is to 1 the better is the prediction capability of our model with 1 being the ideal case.

### Multivariate Regression

Multivariate Regression is a supervised machine learning algorithm involving multiple data variables for analysis. A Multivariate regression is an extension of multiple regression with one dependent variable and multiple independent variables. Based on the number of independent variables, we try to predict the output.


```python
# Initializing the LinearRegression function, training the model and then printing the accuracy scores of the trained model

lr = LinearRegression()
lr.fit(X_train, y_train)
print(f'The score of the Training data is {lr.score(X_train, y_train)}')
print(f'The score of the Testing data is {lr.score(X_test, y_test)}')
y_pred_lr = lr.predict(X_test)
print(f'The mean absolute error percent is {mean_abs_percent_error(y_pred_lr)}')
print(f'The r2 score is {r2_score(y_pred_lr, y_test)}')
```

    The score of the Training data is 0.888632441157034
    The score of the Testing data is 0.8351077574669661
    The mean absolute error percent is 15.068056907229312
    The r2 score is 0.7888165103946227
    

### Random Forest Regression

Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean/average prediction (regression) of the individual trees.


```python
# Initializing the RandomForestRegression function, training the model and then printing the accuracy scores of the trained model

rfr = RandomForestRegressor(random_state = 7)
rfr.fit(X_train, y_train)
print(f'The score of the Training data is {rfr.score(X_train, y_train)}')
print(f'The score of the Testing data is {rfr.score(X_test, y_test)}')
y_pred_rfr = rfr.predict(X_test)
print(f'The mean absolute error percent is {mean_abs_percent_error(y_pred_rfr)}')
print(f'The r2 score is {r2_score(y_pred_rfr, y_test)}')
```

    The score of the Training data is 0.9885352149559811
    The score of the Testing data is 0.9199818535704202
    The mean absolute error percent is 9.406886994353995
    The r2 score is 0.9108685973778583
    

### K-Nearest Neighbor Regression

KNN regression is a non-parametric method that, in an intuitive manner, approximates the association between independent variables and the continuous outcome by averaging the observations in the same neighbourhood. 


```python
# Initializing the kNNRegression function, training the model on a set of parameters and then printing the accuracy scores of the trained model
# thereby infering the best parameter which gives us the best accuracy.

for i in range(1,20):
    print(f'The scores for KNN Regressor with n_neighbours set to {i} is \n')
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    print(f'The score of the Training data is {knn.score(X_train, y_train)}')
    print(f'The score of the Testing data is {knn.score(X_test, y_test)}')
    y_pred_knn = knn.predict(X_test)
    print(f'The mean absolute error percent is {mean_abs_percent_error(y_pred_knn)}')
    print(f'The r2 score is {r2_score(y_pred_knn, y_test)}')
    print()
```

    The scores for KNN Regressor with n_neighbours set to 1 is 
    
    The score of the Training data is 0.9994059244988028
    The score of the Testing data is 0.8059898549598259
    The mean absolute error percent is 17.156387486194177
    The r2 score is 0.7717910992366436
    
    The scores for KNN Regressor with n_neighbours set to 2 is 
    
    The score of the Training data is 0.9545310950365257
    The score of the Testing data is 0.8194391460409256
    The mean absolute error percent is 16.217939765826568
    The r2 score is 0.7764996083751277
    
    The scores for KNN Regressor with n_neighbours set to 3 is 
    
    The score of the Training data is 0.9278804952433335
    The score of the Testing data is 0.817160655414582
    The mean absolute error percent is 15.84978151985848
    The r2 score is 0.7639073824775644
    
    The scores for KNN Regressor with n_neighbours set to 4 is 
    
    The score of the Training data is 0.9141535573644721
    The score of the Testing data is 0.8259057868203146
    The mean absolute error percent is 15.212738173032283
    The r2 score is 0.76483468177751
    
    The scores for KNN Regressor with n_neighbours set to 5 is 
    
    The score of the Training data is 0.897884858453905
    The score of the Testing data is 0.8112628269521057
    The mean absolute error percent is 15.16828407244543
    The r2 score is 0.7332553082003235
    
    The scores for KNN Regressor with n_neighbours set to 6 is 
    
    The score of the Training data is 0.882969430945527
    The score of the Testing data is 0.7973176834861371
    The mean absolute error percent is 15.430706634544142
    The r2 score is 0.7070275695599912
    
    The scores for KNN Regressor with n_neighbours set to 7 is 
    
    The score of the Training data is 0.8750002234903006
    The score of the Testing data is 0.7935527619498413
    The mean absolute error percent is 15.665906581839575
    The r2 score is 0.6967702949979935
    
    The scores for KNN Regressor with n_neighbours set to 8 is 
    
    The score of the Training data is 0.8696478428215536
    The score of the Testing data is 0.7968542536650531
    The mean absolute error percent is 15.753212680169149
    The r2 score is 0.7002538890598943
    
    The scores for KNN Regressor with n_neighbours set to 9 is 
    
    The score of the Training data is 0.864620106793971
    The score of the Testing data is 0.7814832116477056
    The mean absolute error percent is 16.099405089789414
    The r2 score is 0.6662013879469788
    
    The scores for KNN Regressor with n_neighbours set to 10 is 
    
    The score of the Training data is 0.8605787291096589
    The score of the Testing data is 0.7753493465403745
    The mean absolute error percent is 16.178125455918163
    The r2 score is 0.6438686152103142
    
    The scores for KNN Regressor with n_neighbours set to 11 is 
    
    The score of the Training data is 0.8540320825925176
    The score of the Testing data is 0.7697416051344675
    The mean absolute error percent is 16.47806731398199
    The r2 score is 0.6265671804704657
    
    The scores for KNN Regressor with n_neighbours set to 12 is 
    
    The score of the Training data is 0.8475585914039789
    The score of the Testing data is 0.7702970502030979
    The mean absolute error percent is 16.687926956664363
    The r2 score is 0.6277052890946664
    
    The scores for KNN Regressor with n_neighbours set to 13 is 
    
    The score of the Training data is 0.8447461137713198
    The score of the Testing data is 0.7599029281929429
    The mean absolute error percent is 16.93586178031499
    The r2 score is 0.6019193098630518
    
    The scores for KNN Regressor with n_neighbours set to 14 is 
    
    The score of the Training data is 0.8402738215581493
    The score of the Testing data is 0.7566912282969624
    The mean absolute error percent is 17.22086090106902
    The r2 score is 0.5947975690307112
    
    The scores for KNN Regressor with n_neighbours set to 15 is 
    
    The score of the Training data is 0.8363382177210348
    The score of the Testing data is 0.7588486145892734
    The mean absolute error percent is 17.367307591674106
    The r2 score is 0.5984885172731347
    
    The scores for KNN Regressor with n_neighbours set to 16 is 
    
    The score of the Training data is 0.8319763690789683
    The score of the Testing data is 0.7562024525109159
    The mean absolute error percent is 17.496521717167376
    The r2 score is 0.5903314761441927
    
    The scores for KNN Regressor with n_neighbours set to 17 is 
    
    The score of the Training data is 0.8262912211590606
    The score of the Testing data is 0.7544928613560378
    The mean absolute error percent is 17.33871355236679
    The r2 score is 0.583744079219614
    
    The scores for KNN Regressor with n_neighbours set to 18 is 
    
    The score of the Training data is 0.8207615430710955
    The score of the Testing data is 0.7487697725817325
    The mean absolute error percent is 17.409512014481642
    The r2 score is 0.5710590670185473
    
    The scores for KNN Regressor with n_neighbours set to 19 is 
    
    The score of the Training data is 0.816459204271966
    The score of the Testing data is 0.7460066546166357
    The mean absolute error percent is 17.588214333335504
    The r2 score is 0.5685183364105453
    
    


```python
# From the above results we can see that n_neighbors = 2 gives us the best result and so we train our model with n_neighbors = 2

knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_train, y_train)
print(f'The score of the Training data is {knn.score(X_train, y_train)}')
print(f'The score of the Testing data is {knn.score(X_test, y_test)}')
y_pred_knn = knn.predict(X_test)
print(f'The mean absolute error percent is {mean_abs_percent_error(y_pred_knn)}')
print(f'The r2 score is {r2_score(y_pred_knn, y_test)}')
```

    The score of the Training data is 0.9545310950365257
    The score of the Testing data is 0.8194391460409256
    The mean absolute error percent is 16.217939765826568
    The r2 score is 0.7764996083751277
    

### Support Vector Regression

Support Vector Machine (SVM) is a very popular Machine Learning algorithm that is used in both Regression and Classification. Support Vector Regression is similar to Linear Regression in that the equation of the line is y= wx+b 

In SVR, this straight line is referred to as hyperplane. The data points on either side of the hyperplane that are closest to the hyperplane are called Support Vectors which is used to plot the boundary line.

Unlike other Regression models that try to minimize the error between the real and predicted value, the SVR tries to fit the best line within a threshold value (Distance between hyperplane and boundary line)

For a non-linear regression, the kernel function transforms the data to a higher dimensional and performs the linear separation. 

#### Using a linear kernel

The linear kernel, simplest of all kernels, works fine if your dataset if linearly separable.


```python
# Initalizing the grid using GridSearchCV for a linear kernel

grd = GridSearchCV(estimator=SVR(kernel='linear'),
                       param_grid={'C': [0.1, 1, 100, 1000],
                                   'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],},
                       cv=5, verbose=0, n_jobs=-1)
```


```python
# Using the grid initialized before to train our SVR model and print the accuracy results

grid_result = grd.fit(X, y)
best_params = grid_result.best_params_
svm = SVR(kernel='linear', C=best_params["C"], epsilon=best_params["epsilon"], coef0=0.1, shrinking=True,
                   tol=0.001, cache_size=200, verbose=False, max_iter=-1)
svm.fit(X_train, y_train)
print(f'The score of the Training data is {svm.score(X_train, y_train)}')
print(f'The score of the Testing data is {svm.score(X_test, y_test)}')
y_pred_svm = svm.predict(X_test)
print(f'The mean absolute error percent is {mean_abs_percent_error(y_pred_svm)}')
print(f'The r2 score is {r2_score(y_pred_svm, y_test)}')
```

    The score of the Training data is 0.8759269685059281
    The score of the Testing data is 0.8131706639547428
    The mean absolute error percent is 14.140290238073897
    The r2 score is 0.7060728504406104
    

#### Using a polynomial kernel

Polynomial kernel represents the similarity of vectors in training set of data in a feature space over polynomials of the original variables used in kernel.

Polynomial kernels are well suited for problems where all the training data is normalized.


```python
# Initalizing the grid using GridSearchCV for a ploynomial kernel

grd = GridSearchCV(
    estimator=SVR(kernel='poly'),
    param_grid={
        'C': [0.1, 1, 100, 1000],
        'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'degree': [2, 3, 4, 5],
        'coef0': [0.1, 0.01, 0.001, 0.0001]},
    cv=5, verbose=0, n_jobs=-1)
```


```python
# Using the grid initialized before to train our SVR model and print the accuracy results

grid_result = grd.fit(X, y)
best_params = grid_result.best_params_
svm = SVR(kernel='poly', C=best_params["C"], epsilon=best_params["epsilon"], coef0=best_params["coef0"],
               degree=best_params["degree"], shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
svm.fit(X_train, y_train)
print(f'The score of the Training data is {svm.score(X_train, y_train)}')
print(f'The score of the Testing data is {svm.score(X_test, y_test)}')
y_pred_svm = svm.predict(X_test)
print(f'The mean absolute error percent is {mean_abs_percent_error(y_pred_svm)}')
print(f'The r2 score is {r2_score(y_pred_svm, y_test)}')
```

    The score of the Training data is 0.9660990062225835
    The score of the Testing data is 0.9221319732799452
    The mean absolute error percent is 10.633632860092739
    The r2 score is 0.9177117477039606
    

#### Using a Radial Basis Function kernel

RBF uses normal curves around the data points, and sums these so that the decision boundary can be defined by a type of topology condition such as curves where the sum is above a value of 0.5. 


```python
# Initalizing the grid using GridSearchCV for a Radial Basis Function kernel

grd = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        },
        cv=5, verbose=0, n_jobs=-1)
```


```python
# Using the grid initialized before to train our SVR model and print the accuracy results

grid_result = grd.fit(X, y)
best_params = grid_result.best_params_
svm = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                   coef0=0.1, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
svm.fit(X_train, y_train)
print(f'The score of the Training data is {svm.score(X_train, y_train)}')
print(f'The score of the Testing data is {svm.score(X_test, y_test)}')
y_pred_svm = svm.predict(X_test)
print(f'The mean absolute error percent is {mean_abs_percent_error(y_pred_svm)}')
print(f'The r2 score is {r2_score(y_pred_svm, y_test)}')
```

    The score of the Training data is 0.9044412240747342
    The score of the Testing data is 0.8606555567664886
    The mean absolute error percent is 12.168109596045735
    The r2 score is 0.7990505915084811
    

#### Using a Sigmoid Function kernel

Sigmoid kernels de-emphasize extreme correlation. In a way they behave a bit like correlation coefficients, which also has a limited range, emphasizing similarity in orientation. c shifts the operating point on the sigmoid, affecting the relative emphasis of the angle between the inputs.

Sigmoid kernels behave like RBFs for certain parameters. This makes them suited to nonlinear classification/regression.


```python
# Initalizing the grid using GridSearchCV for a Sigmoid kernel

grd = GridSearchCV(
        estimator=SVR(kernel='sigmoid'),
        param_grid={
            'C': [0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'coef0': [0.1, 0.01, 0.001, 0.0001]},
    cv=5, verbose=0, n_jobs=-1)
```


```python
# Using the grid initialized before to train our SVR model and print the accuracy results

grid_result = grd.fit(X, y)
best_params = grid_result.best_params_
svm = SVR(kernel='sigmoid', C=best_params["C"], epsilon=best_params["epsilon"], coef0=best_params["coef0"],
                 shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
svm.fit(X_train, y_train)
print(f'The score of the Training data is {svm.score(X_train, y_train)}')
print(f'The score of the Testing data is {svm.score(X_test, y_test)}')
y_pred_svm = svm.predict(X_test)
print(f'The mean absolute error percent is {mean_abs_percent_error(y_pred_svm)}')
print(f'The r2 score is {r2_score(y_pred_svm, y_test)}')
```

    The score of the Training data is 0.2897352584300895
    The score of the Testing data is 0.26953926387915383
    The mean absolute error percent is 34.092635497181725
    The r2 score is -13.42345659553988
    

### Conclusions

From the above models and results we can arrive at some interesting conclusions:
<br />

<br />

1. We can see that SVR with a polynomial kernel model produces the best result with a r2 score of 91.77% followed by a Random Forest Regressor model which gives us a r2 score of 91.08% both of which signify that our model is good and can be used for predicting the laptop values in real life.
<br />

2. The remaining models that is multivariate regression, kNN Regrssor, SVR using linear kernel and SVR using radial basis function give us a r2 score between 70% to 80% which is also good but usually the models used to predict real world scenarios are expected to have r2 score of more than 80%.
<br />

3. The model trained using SVR with a sigmoid kernel performed the worse with an r2 score of -13 which signifies that the model trained using SVR with a sigmoid kernel is completely unpractical and unusable to predict anything meaningful.
<br />

So, from the above results we can conclude that the model trained using either SVR with Polynomial Kernel or Random Forest Regressor is well suited to use in real world scenarios for predicting the laptop price based on it's features

## Concluding Remarks

1. We have first checked the dataset, understood the type and importance of each and every feature, cleaned all the null or missing values and then made a roadmap of the proper preprocessing technique required to be applied for each feature.
<br />

2. We then went ahead in applying the proper preprocessing for each feature, we then analyzed the feature with the help of bar plots, pie charts and histograms arriving at some meaningful conclusions and statistics which we wrote at the relevant places.
<br />

3. We then went ahead making some additional analysis like price wrt to screen type and such analysis between two different features and also drew a heat-map to see the relationship amongst features.
<br />

4. We then went ahead with training Machine Learning models on our preprocessed dataset, we applied some models such as Multivariate Regression, Random Forest Regression, K-Nearest Neighbor Regressor, and Support Vector Machine with a few different kernels.
<br />

5. We first encoded the data to convert all the categorical data into numerical data suitable for our training our models, then we applied Standardization to help our models understand the data much better and then we drew a heat map to check the relationship between features, so as to help us arrive at a conclusion as to what features to choose to ensure our models learn effectively.
<br />

6. We then trained the models one by one all the while checking the MAPE and r2 score of the trained models and were able to arrive at the conclusion that model trained using SVR with polynomial kernel was the most accurate followed by the model trained using Random Forest Regressor.
<br />

7. The difference between the r2 score of the model trained using SVR with polynomial kernel and that of the model trained using Random Forest Regressor were more or less the same with the difference being only in the 10^-1 range, but if we were to choose between one it would be better to go with Random Forest Regressor than SVR with polynomial kernel since the latter has some problems like 
    a. numerical instability
    b. takes a lot of time if the dataset is large(i.e. time complexity is high/computationally costly)
    c. Scaling is an important fundamental step when working with SVRs otherwise features with higher nominal values will dominate the decision.

    While the former even through it has numerous optimization parameters it's not so easy to make huge mistakes with it unlike SVR's where correct parameters can define the line between misery and victory.
<br />

8. We also saw that SVR with a Sigmoid kernel performed the worse, gave out a negative r2 score and a abnormally high MAPE value which says that our model is unfit for real world use.
<br />

Thus we built a Machine Learning model that was able to predict the laptop prices given the laptop specifications with an accuracy of 90%.
