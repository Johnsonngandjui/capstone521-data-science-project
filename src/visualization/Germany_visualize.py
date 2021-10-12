# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 09:06:03 2021

@author: ngand
"""

import pandas as pd
import matplotlib.pyplot as plt

data= data = pd.read_csv('D:\Senior\Capstone\data-science-enviroment\data\Leagues\Germany_league_V1.csv')

Teams = data.iloc[:,3]
unique_Teams= Teams.unique()

Teams.value_counts().plot(kind='bar');

plt.xlabel('Teams Germany League')
plt.ylabel('Counts')
plt.title('Teams Occurences 1995-2020 ')

plt.show()