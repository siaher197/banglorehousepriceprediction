Bangalore House Price Prediction
During this Bangaluru House Price prediction using Machine Learning tutorial you will learn several things like :-
Exploratory data analysis
Dealing with a missing values or noisy data
Data preprocessing
Create new features from existing features
Remove outliers
Data visualisation
Splitting data into the training and testing 
Train linear regression model and test.
I have trained a Bengaluru House Price prediction model using linear regression algorithm and I got 86% accuracy over the testing data. 

Data Vizualization

Bangalore House Price Prediction
Overview
In this article we are taking very basic problem statement but the features that are involved in this problem statement will have impact on the learning that we will have, so yes! we are taking the Banglore house prediction as our problem statement and our end goal will be to predict the price of the banglore region based on the features that are involved.

Data Description
Columns:

area_type: Type of the area where house is located
availability: Availability of the house in located region
location: Exact location of that house
size: Size of the house
society: Society of the house where it is located
total_sqft: Total square feet of the house
bath: Bathroom in the house
balcony: Balcony in the house
price: Price associated with each flat
Importing Liabraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
Reading the Data from the CSV file
df = pd.read_csv("Bengaluru_House_Data.csv")
df.head()
area_type	availability	location	size	society	total_sqft	bath	balcony	price
0	Super built-up Area	19-Dec	Electronic City Phase II	2 BHK	Coomee	1056	2.0	1.0	39.07
1	Plot Area	Ready To Move	Chikka Tirupathi	4 Bedroom	Theanmp	2600	5.0	3.0	120.00
2	Built-up Area	Ready To Move	Uttarahalli	3 BHK	NaN	1440	2.0	3.0	62.00
3	Super built-up Area	Ready To Move	Lingadheeranahalli	3 BHK	Soiewre	1521	3.0	1.0	95.00
4	Super built-up Area	Ready To Move	Kothanur	2 BHK	NaN	1200	2.0	1.0	51.00
# Printing the shape of the dataframe
df.shape
(13320, 9)
Exploratory Data Analysis (EDA)
# Let's have a look on all the columns in our dataset
df.columns
Index(['area_type', 'availability', 'location', 'size', 'society',
       'total_sqft', 'bath', 'balcony', 'price'],
      dtype='object')
# Information about the dataset
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 13320 entries, 0 to 13319
Data columns (total 9 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   area_type     13320 non-null  object 
 1   availability  13320 non-null  object 
 2   location      13319 non-null  object 
 3   size          13304 non-null  object 
 4   society       7818 non-null   object 
 5   total_sqft    13320 non-null  object 
 6   bath          13247 non-null  float64
 7   balcony       12711 non-null  float64
 8   price         13320 non-null  float64
dtypes: float64(3), object(6)
memory usage: 936.7+ KB
# To know more about the dataset
df.describe()
bath	balcony	price
count	13247.000000	12711.000000	13320.000000
mean	2.692610	1.584376	112.565627
std	1.341458	0.817263	148.971674
min	1.000000	0.000000	8.000000
25%	2.000000	1.000000	50.000000
50%	2.000000	2.000000	72.000000
75%	3.000000	2.000000	120.000000
max	40.000000	3.000000	3600.000000
# Now with the help of is null function we will check the number of null values in our dataset
df.isnull().head()
area_type	availability	location	size	society	total_sqft	bath	balcony	price
0	False	False	False	False	False	False	False	False	False
1	False	False	False	False	False	False	False	False	False
2	False	False	False	False	True	False	False	False	False
3	False	False	False	False	False	False	False	False	False
4	False	False	False	False	True	False	False	False	False
# Now with the help of is null function we will check the number of null values in our dataset
df.isnull().sum()
area_type          0
availability       0
location           1
size              16
society         5502
total_sqft         0
bath              73
balcony          609
price              0
dtype: int64
# Here we will be using group by function to group up the area_type field
df.groupby("area_type")["area_type"].agg("count")
area_type
Built-up  Area          2418
Carpet  Area              87
Plot  Area              2025
Super built-up  Area    8790
Name: area_type, dtype: int64
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 13320 entries, 0 to 13319
Data columns (total 9 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   area_type     13320 non-null  object 
 1   availability  13320 non-null  object 
 2   location      13319 non-null  object 
 3   size          13304 non-null  object 
 4   society       7818 non-null   object 
 5   total_sqft    13320 non-null  object 
 6   bath          13247 non-null  float64
 7   balcony       12711 non-null  float64
 8   price         13320 non-null  float64
dtypes: float64(3), object(6)
memory usage: 936.7+ KB
df.head()
area_type	availability	location	size	society	total_sqft	bath	balcony	price
0	Super built-up Area	19-Dec	Electronic City Phase II	2 BHK	Coomee	1056	2.0	1.0	39.07
1	Plot Area	Ready To Move	Chikka Tirupathi	4 Bedroom	Theanmp	2600	5.0	3.0	120.00
2	Built-up Area	Ready To Move	Uttarahalli	3 BHK	NaN	1440	2.0	3.0	62.00
3	Super built-up Area	Ready To Move	Lingadheeranahalli	3 BHK	Soiewre	1521	3.0	1.0	95.00
4	Super built-up Area	Ready To Move	Kothanur	2 BHK	NaN	1200	2.0	1.0	51.00
# Dropping less important features
df = df.drop(["area_type", "society","balcony", "availability"], axis = "columns")
df.shape
(13320, 5)
# Dropping null values
df = df.dropna()
df.isnull().sum()
location      0
size          0
total_sqft    0
bath          0
price         0
dtype: int64
df.shape
(13246, 5)
Feature Engineering
# Now, here on the size column we will be using the unique function to see all the distinct size of the houses.
df["size"].unique()
array(['2 BHK', '4 Bedroom', '3 BHK', '4 BHK', '6 Bedroom', '3 Bedroom',
       '1 BHK', '1 RK', '1 Bedroom', '8 Bedroom', '2 Bedroom',
       '7 Bedroom', '5 BHK', '7 BHK', '6 BHK', '5 Bedroom', '11 BHK',
       '9 BHK', '9 Bedroom', '27 BHK', '10 Bedroom', '11 Bedroom',
       '10 BHK', '19 BHK', '16 BHK', '43 Bedroom', '14 BHK', '8 BHK',
       '12 Bedroom', '13 BHK', '18 Bedroom'], dtype=object)
From the above output it is clear that there is either bedroom or BHK. One is BHK and the other one is Bedroom amd we know that this string data (BHK and Bedroom) will hamper the data so we will be removing this object type data and change the type to integer.

df['BHK'] = df["size"].apply(lambda x: int(x.split(" ")[0]))
df.head()
location	size	total_sqft	bath	price	BHK
0	Electronic City Phase II	2 BHK	1056	2.0	39.07	2
1	Chikka Tirupathi	4 Bedroom	2600	5.0	120.00	4
2	Uttarahalli	3 BHK	1440	2.0	62.00	3
3	Lingadheeranahalli	3 BHK	1521	3.0	95.00	3
4	Kothanur	2 BHK	1200	2.0	51.00	2
df.total_sqft.unique()
array(['1056', '2600', '1440', ..., '1133 - 1384', '774', '4689'],
      dtype=object)
# Exploring total_sqft feature
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
df[~df["total_sqft"].apply(is_float)].head(10)
location	size	total_sqft	bath	price	BHK
30	Yelahanka	4 BHK	2100 - 2850	4.0	186.000	4
122	Hebbal	4 BHK	3067 - 8156	4.0	477.000	4
137	8th Phase JP Nagar	2 BHK	1042 - 1105	2.0	54.005	2
165	Sarjapur	2 BHK	1145 - 1340	2.0	43.490	2
188	KR Puram	2 BHK	1015 - 1540	2.0	56.800	2
410	Kengeri	1 BHK	34.46Sq. Meter	1.0	18.500	1
549	Hennur Road	2 BHK	1195 - 1440	2.0	63.770	2
648	Arekere	9 Bedroom	4125Perch	9.0	265.000	9
661	Yelahanka	2 BHK	1120 - 1145	2.0	48.130	2
672	Bettahalsoor	4 Bedroom	3090 - 5002	4.0	445.000	4
From the above output we can see that that total sq. ft. column is a range type of data (example: 2100-2850) which in statistics terms also known as the confidence intervals (confidence intervals are derived from the point estimates) there are also other cases where value is there along with the unit too (example 34.46 sq. meter). so for that reason only we will be removing the end cases.

def convert_sqft_to_number(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df = df.copy()
df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_number)
df.head(10)
location	size	total_sqft	bath	price	BHK
0	Electronic City Phase II	2 BHK	1056.0	2.0	39.07	2
1	Chikka Tirupathi	4 Bedroom	2600.0	5.0	120.00	4
2	Uttarahalli	3 BHK	1440.0	2.0	62.00	3
3	Lingadheeranahalli	3 BHK	1521.0	3.0	95.00	3
4	Kothanur	2 BHK	1200.0	2.0	51.00	2
5	Whitefield	2 BHK	1170.0	2.0	38.00	2
6	Old Airport Road	4 BHK	2732.0	4.0	204.00	4
7	Rajaji Nagar	4 BHK	3300.0	4.0	600.00	4
8	Marathahalli	3 BHK	1310.0	3.0	63.25	3
9	Gandhi Bazar	6 Bedroom	1020.0	6.0	370.00	6
Here for better understanding we are inculcating another column in the dataset which is "price per sq. ft."

df = df.copy()
df["price_per_sqft"] = df["price"]*100000/df["total_sqft"]
df.head()
location	size	total_sqft	bath	price	BHK	price_per_sqft
0	Electronic City Phase II	2 BHK	1056.0	2.0	39.07	2	3699.810606
1	Chikka Tirupathi	4 Bedroom	2600.0	5.0	120.00	4	4615.384615
2	Uttarahalli	3 BHK	1440.0	2.0	62.00	3	4305.555556
3	Lingadheeranahalli	3 BHK	1521.0	3.0	95.00	3	6245.890861
4	Kothanur	2 BHK	1200.0	2.0	51.00	2	4250.000000
So here for the reduction of number of location we have to use dimensionality reduction method in the case of the categorical values.

df.location = df.location.apply(lambda x: x.strip())
location_stats = df['location'].value_counts(ascending=False)
location_stats
Whitefield                                        535
Sarjapur  Road                                    392
Electronic City                                   304
Kanakpura Road                                    266
Thanisandra                                       236
                                                 ... 
Kamdhenu Nagar                                      1
S R Layout                                          1
6th block banashankari 3rd stage, 100 feet ORR      1
Junnasandra                                         1
Doddanakunte                                        1
Name: location, Length: 1293, dtype: int64
len(location_stats[location_stats<=10])
1052
location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10
BTM 1st Stage                                     10
Gunjur Palya                                      10
Dodsworth Layout                                  10
1st Block Koramangala                             10
Ganga Nagar                                       10
                                                  ..
Kamdhenu Nagar                                     1
S R Layout                                         1
6th block banashankari 3rd stage, 100 feet ORR     1
Junnasandra                                        1
Doddanakunte                                       1
Name: location, Length: 1052, dtype: int64
df.location = df.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df.location.unique())
242
df.head()
location	size	total_sqft	bath	price	BHK	price_per_sqft
0	Electronic City Phase II	2 BHK	1056.0	2.0	39.07	2	3699.810606
1	Chikka Tirupathi	4 Bedroom	2600.0	5.0	120.00	4	4615.384615
2	Uttarahalli	3 BHK	1440.0	2.0	62.00	3	4305.555556
3	Lingadheeranahalli	3 BHK	1521.0	3.0	95.00	3	6245.890861
4	Kothanur	2 BHK	1200.0	2.0	51.00	2	4250.000000
Here we will discard some more data. Because, normally if a square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier. We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft

df[df.total_sqft/df.BHK<300].head()
location	size	total_sqft	bath	price	BHK	price_per_sqft
9	other	6 Bedroom	1020.0	6.0	370.0	6	36274.509804
45	HSR Layout	8 Bedroom	600.0	9.0	200.0	8	33333.333333
58	Murugeshpalya	6 Bedroom	1407.0	4.0	150.0	6	10660.980810
68	Devarachikkanahalli	8 Bedroom	1350.0	7.0	85.0	8	6296.296296
70	other	3 Bedroom	500.0	3.0	100.0	3	20000.000000
df = df[~(df.total_sqft/df.BHK<300)]
df.shape
(12502, 7)
df.describe()
total_sqft	bath	price	BHK	price_per_sqft
count	12456.000000	12502.000000	12502.000000	12502.000000	12456.000000
mean	1590.189927	2.564790	111.311915	2.650696	6308.502826
std	1260.404795	1.084946	152.089966	0.981698	4168.127339
min	300.000000	1.000000	9.000000	1.000000	267.829813
25%	1115.000000	2.000000	49.000000	2.000000	4210.526316
50%	1300.000000	2.000000	70.000000	3.000000	5294.117647
75%	1700.000000	3.000000	115.000000	3.000000	6916.666667
max	52272.000000	16.000000	3600.000000	16.000000	176470.588235
Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices. We should remove outliers per location using mean and one Standard Deviation

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df = remove_pps_outliers(df)
df.shape
(10241, 7)
df.head()
location	size	total_sqft	bath	price	BHK	price_per_sqft
0	1st Block Jayanagar	4 BHK	2850.0	4.0	428.0	4	15017.543860
1	1st Block Jayanagar	3 BHK	1630.0	3.0	194.0	3	11901.840491
2	1st Block Jayanagar	3 BHK	1875.0	2.0	235.0	3	12533.333333
3	1st Block Jayanagar	3 BHK	1200.0	2.0	130.0	3	10833.333333
4	1st Block Jayanagar	2 BHK	1235.0	2.0	148.0	2	11983.805668
Data Vizualization
df.head(10)
location	size	total_sqft	bath	price	BHK	price_per_sqft
0	1st Block Jayanagar	4 BHK	2850.0	4.0	428.0	4	15017.543860
1	1st Block Jayanagar	3 BHK	1630.0	3.0	194.0	3	11901.840491
2	1st Block Jayanagar	3 BHK	1875.0	2.0	235.0	3	12533.333333
3	1st Block Jayanagar	3 BHK	1200.0	2.0	130.0	3	10833.333333
4	1st Block Jayanagar	2 BHK	1235.0	2.0	148.0	2	11983.805668
5	1st Block Jayanagar	4 BHK	2750.0	4.0	413.0	4	15018.181818
6	1st Block Jayanagar	4 BHK	2450.0	4.0	368.0	4	15020.408163
7	1st Phase JP Nagar	4 BHK	2825.0	4.0	250.0	4	8849.557522
8	1st Phase JP Nagar	3 BHK	1875.0	3.0	167.0	3	8906.666667
9	1st Phase JP Nagar	5 Bedroom	1500.0	5.0	85.0	5	5666.666667
# Ploting the Scatter Chart for 2 BHK and 3 BHK properties
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.BHK==2)]
    bhk3 = df[(df.location==location) & (df.BHK==3)]
    matplotlib.rcParams['figure.figsize'] = (8,6)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df,"1st Phase JP Nagar")

# Ploting the Scatter Chart for 3 BHK and 4 BHK properties
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.BHK==3)]
    bhk3 = df[(df.location==location) & (df.BHK==4)]
    matplotlib.rcParams['figure.figsize'] = (8,6)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='3 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='4 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df,"1st Block Jayanagar")

# Ploting the histogram for Price Per Square Feet vs Count
plt.hist(df.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
Text(0, 0.5, 'Count')

# Ploting the histogram for Number of bathrooms vs Count
plt.hist(df.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
Text(0, 0.5, 'Count')

df[df.bath>10]
location	size	total_sqft	bath	price	BHK	price_per_sqft
5277	Neeladri Nagar	10 BHK	4000.0	12.0	160.0	10	4000.000000
8486	other	10 BHK	12000.0	12.0	525.0	10	4375.000000
8575	other	16 BHK	10000.0	16.0	550.0	16	5500.000000
9308	other	11 BHK	6000.0	12.0	150.0	11	2500.000000
9639	other	13 BHK	5425.0	13.0	275.0	13	5069.124424
It is unusual to have 2 more bathrooms than number of bedrooms in a home. So we are discarding that also.

df[df.bath>df.BHK+2]
location	size	total_sqft	bath	price	BHK	price_per_sqft
1626	Chikkabanavar	4 Bedroom	2460.0	7.0	80.0	4	3252.032520
5238	Nagasandra	4 Bedroom	7000.0	8.0	450.0	4	6428.571429
6711	Thanisandra	3 BHK	1806.0	6.0	116.0	3	6423.034330
8411	other	6 BHK	11338.0	9.0	1000.0	6	8819.897689
df.head()
location	size	total_sqft	bath	price	BHK	price_per_sqft
0	1st Block Jayanagar	4 BHK	2850.0	4.0	428.0	4	15017.543860
1	1st Block Jayanagar	3 BHK	1630.0	3.0	194.0	3	11901.840491
2	1st Block Jayanagar	3 BHK	1875.0	2.0	235.0	3	12533.333333
3	1st Block Jayanagar	3 BHK	1200.0	2.0	130.0	3	10833.333333
4	1st Block Jayanagar	2 BHK	1235.0	2.0	148.0	2	11983.805668
df.shape
(10241, 7)
Using One Hot Encoding for Location
dummies = pd.get_dummies(df.location)
dummies.head(10)
1st Block Jayanagar	1st Phase JP Nagar	2nd Phase Judicial Layout	2nd Stage Nagarbhavi	5th Block Hbr Layout	5th Phase JP Nagar	6th Phase JP Nagar	7th Phase JP Nagar	8th Phase JP Nagar	9th Phase JP Nagar	...	Vishveshwarya Layout	Vishwapriya Layout	Vittasandra	Whitefield	Yelachenahalli	Yelahanka	Yelahanka New Town	Yelenahalli	Yeshwanthpur	other
0	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
6	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
7	0	1	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
8	0	1	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
9	0	1	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
10 rows × 242 columns

Concatinating both the dataframes together
df = pd.concat([df,dummies.drop('other',axis='columns')],axis='columns')
df.head()
location	size	total_sqft	bath	price	BHK	price_per_sqft	1st Block Jayanagar	1st Phase JP Nagar	2nd Phase Judicial Layout	...	Vijayanagar	Vishveshwarya Layout	Vishwapriya Layout	Vittasandra	Whitefield	Yelachenahalli	Yelahanka	Yelahanka New Town	Yelenahalli	Yeshwanthpur
0	1st Block Jayanagar	4 BHK	2850.0	4.0	428.0	4	15017.543860	1	0	0	...	0	0	0	0	0	0	0	0	0	0
1	1st Block Jayanagar	3 BHK	1630.0	3.0	194.0	3	11901.840491	1	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1st Block Jayanagar	3 BHK	1875.0	2.0	235.0	3	12533.333333	1	0	0	...	0	0	0	0	0	0	0	0	0	0
3	1st Block Jayanagar	3 BHK	1200.0	2.0	130.0	3	10833.333333	1	0	0	...	0	0	0	0	0	0	0	0	0	0
4	1st Block Jayanagar	2 BHK	1235.0	2.0	148.0	2	11983.805668	1	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 248 columns

df = df.drop('location',axis='columns')
df.head()
size	total_sqft	bath	price	BHK	price_per_sqft	1st Block Jayanagar	1st Phase JP Nagar	2nd Phase Judicial Layout	2nd Stage Nagarbhavi	...	Vijayanagar	Vishveshwarya Layout	Vishwapriya Layout	Vittasandra	Whitefield	Yelachenahalli	Yelahanka	Yelahanka New Town	Yelenahalli	Yeshwanthpur
0	4 BHK	2850.0	4.0	428.0	4	15017.543860	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	3 BHK	1630.0	3.0	194.0	3	11901.840491	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	3 BHK	1875.0	2.0	235.0	3	12533.333333	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	3 BHK	1200.0	2.0	130.0	3	10833.333333	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	2 BHK	1235.0	2.0	148.0	2	11983.805668	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 247 columns

X = df.drop(['price'],axis='columns')
X.head()
size	total_sqft	bath	BHK	price_per_sqft	1st Block Jayanagar	1st Phase JP Nagar	2nd Phase Judicial Layout	2nd Stage Nagarbhavi	5th Block Hbr Layout	...	Vijayanagar	Vishveshwarya Layout	Vishwapriya Layout	Vittasandra	Whitefield	Yelachenahalli	Yelahanka	Yelahanka New Town	Yelenahalli	Yeshwanthpur
0	4 BHK	2850.0	4.0	4	15017.543860	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	3 BHK	1630.0	3.0	3	11901.840491	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	3 BHK	1875.0	2.0	3	12533.333333	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	3 BHK	1200.0	2.0	3	10833.333333	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	2 BHK	1235.0	2.0	2	11983.805668	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 246 columns

X = df.drop(['size'],axis='columns')
X.head()
total_sqft	bath	price	BHK	price_per_sqft	1st Block Jayanagar	1st Phase JP Nagar	2nd Phase Judicial Layout	2nd Stage Nagarbhavi	5th Block Hbr Layout	...	Vijayanagar	Vishveshwarya Layout	Vishwapriya Layout	Vittasandra	Whitefield	Yelachenahalli	Yelahanka	Yelahanka New Town	Yelenahalli	Yeshwanthpur
0	2850.0	4.0	428.0	4	15017.543860	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	1630.0	3.0	194.0	3	11901.840491	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1875.0	2.0	235.0	3	12533.333333	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	1200.0	2.0	130.0	3	10833.333333	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	1235.0	2.0	148.0	2	11983.805668	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 246 columns

y = df.price
y.head()
0    428.0
1    194.0
2    235.0
3    130.0
4    148.0
Name: price, dtype: float64
X = X.drop(['price_per_sqft'],axis='columns')
X.head()
total_sqft	bath	price	BHK	1st Block Jayanagar	1st Phase JP Nagar	2nd Phase Judicial Layout	2nd Stage Nagarbhavi	5th Block Hbr Layout	5th Phase JP Nagar	...	Vijayanagar	Vishveshwarya Layout	Vishwapriya Layout	Vittasandra	Whitefield	Yelachenahalli	Yelahanka	Yelahanka New Town	Yelenahalli	Yeshwanthpur
0	2850.0	4.0	428.0	4	1	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	1630.0	3.0	194.0	3	1	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1875.0	2.0	235.0	3	1	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	1200.0	2.0	130.0	3	1	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	1235.0	2.0	148.0	2	1	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 245 columns

X = X.drop(['price'],axis='columns')
X.head()
total_sqft	bath	BHK	1st Block Jayanagar	1st Phase JP Nagar	2nd Phase Judicial Layout	2nd Stage Nagarbhavi	5th Block Hbr Layout	5th Phase JP Nagar	6th Phase JP Nagar	...	Vijayanagar	Vishveshwarya Layout	Vishwapriya Layout	Vittasandra	Whitefield	Yelachenahalli	Yelahanka	Yelahanka New Town	Yelenahalli	Yeshwanthpur
0	2850.0	4.0	4	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	1630.0	3.0	3	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1875.0	2.0	3	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	1200.0	2.0	3	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	1235.0	2.0	2	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 244 columns

X.shape
(10241, 244)
y.shape
(10241,)
Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)
0.7900425477740949
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)
array([0.77907697, 0.7535079 , 0.79892911, 0.80972959, 0.8025454 ])
Here we are using Grid Search CV for 3 different types of Regression models.

Linear Regression
Lasso Regression
Decision Tree Regression
Model Building
def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])
Model Evaluation
find_best_model_using_gridsearchcv(X,y)
model	best_score	best_params
0	linear_regression	0.788758	{'normalize': False}
1	lasso	0.656938	{'alpha': 1, 'selection': 'cyclic'}
2	decision_tree	0.683803	{'criterion': 'friedman_mse', 'splitter': 'ran...
Model Testing
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0])
Here we are predicting the house prices based on Location, Size, Bathroom, and BHK

predict_price('1st Phase JP Nagar',1000, 2, 2)
87.81400704680102
df.head()
size	total_sqft	bath	price	BHK	price_per_sqft	1st Block Jayanagar	1st Phase JP Nagar	2nd Phase Judicial Layout	2nd Stage Nagarbhavi	...	Vijayanagar	Vishveshwarya Layout	Vishwapriya Layout	Vittasandra	Whitefield	Yelachenahalli	Yelahanka	Yelahanka New Town	Yelenahalli	Yeshwanthpur
0	4 BHK	2850.0	4.0	428.0	4	15017.543860	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	3 BHK	1630.0	3.0	194.0	3	11901.840491	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	3 BHK	1875.0	2.0	235.0	3	12533.333333	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	3 BHK	1200.0	2.0	130.0	3	10833.333333	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	2 BHK	1235.0	2.0	148.0	2	11983.805668	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 247 columns

predict_price('Banashankari Stage V',2000, 3, 3)
99.91478843038549
predict_price('2nd Stage Nagarbhavi',5000, 2, 2)
476.63534892400094
predict_price('Indira Nagar',1500, 3, 3)
204.0517711620694
Conclusion
From all the above models, we can clearly say that Linear Regression perform best for this dataset.
