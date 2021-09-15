import pandas as pd

data= pd.read_csv("D:\Senior\Capstone\data-science-enviroment\data\BIG FIVE 1995-2019.csv")

missing = data.isna().sum()

print(missing)
#0 missing data, this is great!