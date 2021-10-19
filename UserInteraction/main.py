import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import pandas as pd

st.title('Who will win?')

st.write("""
# Explore different classifier and datasets
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('England_2019', 'Breast Cancer', 'Wine')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)
#data set
def get_dataset(name):
    data = None
    if name == 'England_2019':
        data = pd.read_csv('D:/Senior/Capstone/data-science-enviroment/data/2019/England_2019.csv')
        data= data.drop(columns=['Date','Country','Year'])
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.iloc[:,[0,1,2,4,7]].values
# Creating Output : All the dependent variables
    y = data.iloc[:,-1].values
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))
st.write( X)

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

st.sidebar.write("""# Make a prediction""")
st.sidebar.write("Use")
round= st.sidebar.slider('What round did these teams last play in?', 1, 42)
home_team= st.sidebar.slider('Home Team (use key at the bottom of the page)', 0, 225) 
away_team= st.sidebar.slider('Away Team (use key at the bottom of the page)', 0, 225)
away_ft_score= st.sidebar.slider('Full Time Score of Away Team', 0, 10)
ggd= st.sidebar.slider('Goal Game Difference', 0, 10)


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'],)
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#optimization (scaling)
from sklearn.preprocessing import StandardScaler
standardScaler_X = StandardScaler()
X_train = standardScaler_X.fit_transform(X_train)
X_test = standardScaler_X.transform(X_test)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

predict = (clf.predict([[round,home_team,away_team,away_ft_score,ggd]]))

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
if predict ==1 :
    st.write('Predicted Game: Home Team won')
elif predict == 2:
    st.write('Predicted Game: Away Team won')
else:
    st.write('Predicted Game: Draw')
#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(5)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
x3 = X_projected[:, 2]
x4 = X_projected[:, 3]
x5 = X_projected[:, 4]

fig = plt.figure()
plt.scatter( x1, x5,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Rounds ')
plt.ylabel('GDD ')
plt.colorbar()

#plt.show()
st.pyplot(fig)

st.title('Key')
st.text('Match the index with your team starting at 0. \nFor better prediction, use the dataset(country) at least one of your team is in')
team_key = pd.read_csv('D:/Senior/Capstone/data-science-enviroment/data/Teams_1.csv')
st.write(team_key.iloc[:,[1]].values)
