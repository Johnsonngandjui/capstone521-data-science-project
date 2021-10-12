import pandas as pd
import numpy as np

data= data = pd.read_csv('D:\Senior\Capstone\data-science-enviroment\data\BIG FIVE 1995-2019.csv')
data.dtypes

data.columns = data.columns.str.replace(' ','_')

#redundent information, FT/HT Team 1 or 2 to get the following information
data= data.drop(columns=['FT', 'HT'])
for col in data.columns:
    print(col)

#define conditions
conditions = [data['FT_Team_1'] > data['FT_Team_2'], 
              data['FT_Team_1'] < data['FT_Team_2']]

#1= homewin 2=homelost
options = ['1', '2']

#create new column in DataFrame that displays results of comparisons, 3= tie
data['Outcome'] = np.select(conditions, options, default='3')


#transforming countries from categorical to numerical
unique_values_country = data.Country.unique()
X = data.iloc[:,:].values
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,5] = labelencoder_X.fit_transform(X[:,5])
Z = pd.DataFrame(X)

Z.columns =['Round', 'Date', 'Team_1', 'Team_2', 'Year', 'Country', 'FT_Team_1', 'FT_Team_2', 'HT_Team_1', 'HT_Team_2', 'GGD', 'Team_1_(pts)', 'Team_2_(pts)', 'Outcome' ]
 
#Z.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Preprocessed_V1.csv', index=False)  

England = Z.loc[Z['Country'] == 1]
England.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Leagues/England_league_V1.csv', index=False)

Spain = Z.loc[Z['Country'] == 1]
Spain.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Leagues/Spain_league_V1.csv', index=False)

France = Z.loc[Z['Country'] == 2]
France.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Leagues/France_league_V1.csv', index=False)

Germany = Z.loc[Z['Country'] == 3]
Germany.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Leagues/Germany_league_V1.csv', index=False)

Italy = Z.loc[Z['Country'] == 4]
Italy.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Leagues/Italy_league_V1.csv', index=False)