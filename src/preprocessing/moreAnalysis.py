# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:25:26 2021

@author: ngand
"""

import pandas as pd

data = pd.read_csv('D:\Senior\Capstone\data-science-enviroment\data\Leagues\England_league_V1.csv')

#extracting england 2019 season
england_2019_season = data.loc[data['Year'] == 2019]
england_2019_season.to_csv('D:/Senior/Capstone/data-science-enviroment/data/2019/England_2019.csv', index=False)


#avg hometeam goals
avg_goals_home= data['FT_Team_1'].sum()/data['FT_Team_1'].count()
print('avg goals score when at home',avg_goals_home)
#avg awayteam goals
avg_goals_away= data['FT_Team_2'].sum()/data['FT_Team_2'].count()
print('avg goals score when at away',avg_goals_away)


#hometeam win 3 more goals. 
homewin_3_more= data.loc[(data['FT_Team_1'] >= 3) & (data['Outcome'] ==1)]
homewin_3_more_count= homewin_3_more['FT_Team_1'].count()
print('home team wins which includes 3 or more goals being scored', homewin_3_more_count)
#hometeam win
homewin= data.loc[(data['Outcome'] ==1)]
homewin_count= homewin['FT_Team_1'].count()
print('total home team wins', homewin_count)

#percentage of the home victories 3 or more goals have been scored
print('In the' , (homewin_3_more_count/homewin_count)*100,'% of the home victories 3 or more goals have been scored \n' )

#draw 3 more goals. 
Draw_3_more= data.loc[(data['FT_Team_1'] >= 3) & (data['Outcome'] ==3)]
Draw_3_more_count= Draw_3_more['FT_Team_1'].count()
print('Draws which includes 3 or more goals being scored', Draw_3_more_count)

#hometeam win
Draw= data.loc[(data['Outcome'] ==3)]
Draw_count= Draw['FT_Team_1'].count()
print('total Draw', Draw_count)

#percentage of the draw victories 3 or more goals have been scored
print('In the' , (Draw_3_more_count/Draw_count)*100,'% of the draws 3 or more goals have been scored \n' )

#awayteam win 3 more goals. 
awaywin_3_more= data.loc[(data['FT_Team_2'] >= 3) & (data['Outcome'] ==2)]
awaywin_3_more_count= awaywin_3_more['FT_Team_1'].count()
print('away team wins which includes 3 or more goals being scored', awaywin_3_more_count)

#awayteam win
awaywin= data.loc[(data['Outcome'] ==2)]
awaywin_count= awaywin['FT_Team_1'].count()
print('total home team wins', awaywin_count)

#percentage of the home victories 3 or more goals have been scored
print('In the' , (awaywin_3_more_count/awaywin_count)*100,'% of the away victories 3 or more goals have been scored \n' )


victory3goals= (((homewin_3_more_count/homewin_count)*100))
denom= (((homewin_3_more_count/homewin_count)*100)+((Draw_3_more_count/Draw_count)*100)+((awaywin_3_more_count/awaywin_count)*100))
solution= (victory3goals/denom)
print(solution*100)


