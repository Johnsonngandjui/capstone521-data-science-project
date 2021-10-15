import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display

data = pd.read_csv('D:\Senior\Capstone\data-science-enviroment\data\Leagues\England_league_V1.csv')
data= data.drop(columns=['Date'])
display(data.head())

# Total number of matches.
n_matches = data.shape[0]

# Calculate number of features. -1 because we are saving one as the target variable (win/lose/draw)
n_features = data.shape[1] - 1

# Calculate matches won by home team.
n_homewins = len(data[data.Outcome == 1])

# Calculate win rate for home team.
win_rate = (float(n_homewins) / (n_matches)) * 100

print( "Total number of matches: {}".format(n_matches))
print ("Number of features: {}".format(n_features))
print ("Number of matches won by home team: {}".format(n_homewins))
print ("Win rate of home team: {:.2f}%".format(win_rate))

#visualize the distribution of data
#shows correlation between the different features and we want to use
#Scatter plots show how much one variable is affected by another. 

from pandas.plotting import scatter_matrix
scatter_matrix(data[['FT_Team_1','FT_Team_2','GGD','Team_1_(pts)','Team_2_(pts)','Outcome']], figsize=(10,10))

# Separate into feature set and target variable
#FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
X_all = data.drop(['Outcome'],1)
y_all = data['Outcome']

# Standardising the data.
from sklearn.preprocessing import scale

#Center to the mean and component wise scale to unit variance.
cols = [['FT_Team_1','FT_Team_2','GGD','Team_1_(pts)','Team_2_(pts)']]
for col in cols:
    X_all[col] = scale(X_all[col])
    

