# My Data Science Capstone enviroment

This is a help me read me file form informtion of what is in this project.

Installation
You need these dependencies:

pip install streamlit
pip install scikit-learn
pip install matplotlibgit
pip install XGBoost
Usage

Run
navigate to Userinteraction Folder
streamlit run main.py

To terminate spark session on jupyter notebook
run pyspark in file directory to start jupyter notebook with spark

session = SparkSession.builder.appName("myApp").getOrCreate()
session.stop()
