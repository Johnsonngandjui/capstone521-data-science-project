import pandas as pd

data= data = pd.read_csv('D:\Senior\Capstone\data-science-enviroment\data\BIG FIVE 1995-2019.csv')
data.dtypes

data.columns = data.columns.str.replace(' ','_')

#redundent information, FT/HT Team 1 or 2 to get the following information
data= data.drop(columns=['FT', 'HT'])
for col in data.columns:
    print(col)

#transforming countries from categorical to numerical
unique_values_country = data.Country.unique()
X = data.iloc[:,:].values


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,5] = labelencoder_X.fit_transform(X[:,5])
Z = pd.DataFrame(X)



#Z.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Preprocessed_V1.csv', index=False)  

England = Z.loc[Z[5] == 0]
England.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Leagues/England_league_V1.csv', index=False)

Spain = Z.loc[Z[5] == 1]
Spain.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Leagues/Spain_league_V1.csv', index=False)

France = Z.loc[Z[5] == 2]
France.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Leagues/France_league_V1.csv', index=False)

Germany = Z.loc[Z[5] == 3]
Germany.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Leagues/Germany_league_V1.csv', index=False)

Italy = Z.loc[Z[5] == 4]
Italy.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Leagues/Italy_league_V1.csv', index=False)