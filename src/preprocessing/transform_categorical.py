import pandas as pd

data= data = pd.read_csv('D:\Senior\Capstone\data-science-enviroment\data\BIG FIVE 1995-2019.csv')
data.dtypes

#redundent information, FT/HT Team 1 or 2 to get the following information
data= data.drop(columns=['FT', 'HT'])
for col in data.columns:
    print(col)

#transforming countries from categorical to numerical
unique_values = data.Country.unique()
X = data.iloc[:,:].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,5] = labelencoder_X.fit_transform(X[:,5])
Z = pd.DataFrame(X)

#Z.to_csv('D:/Senior/Capstone/data-science-enviroment/data/Preprocessed_V1.csv')
