# DSND-Term2-P1-Disaster-Response-Pipelines
Udacity Data Scientist Nanodegree - Term 2 - Project 1 - Disaster Response Pipelines

### Table of Contents

1. [Project Overview](#ProjectOverview)
2. [Project Components](#ProjectComponents)
3. [File Description](#FileDescription)
4. [Requirements](#Requirements)
5. [Instructions](#Instructions)

## 1. Project Overview <a name="ProjectOverview"></a>
The goal of the project is to apply data engineering skills to analyze the [Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data/) dataset provided by Figure Eight, and build a ML model classifying disaster messages based on predefined categories. End result of this project is a web application where an emergency worker can type in input message and our application using ML model in the background, classifies the message based on several categories and then highlights various categories that the input message belong to.

### 2. Project Components <a name="ProjectComponents"></a>
There are three components:

### 2.1. ETL Pipeline
This is a data cleaning pipeline, that:
- Load Messages and Categories csv datasets 
- Merge the two datasets
- Clean the data
- Save data into a SQLite database

### 2.2. ML Pipeline
This is a machine learning pipeline, that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### 2.3. Flask Web App
A Web application that is built and run using Flask framework.

## 3. File Description <a name="FileDescription"></a>
There are three main folders:

1. data
    - disaster_categories.csv: dataset including all the categories
    - disaster_messages.csv: dataset including all the messages
    - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
    - DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
    - train_classifier.py: machine learning pipeline scripts to train and export a classifier
    - classifier.pkl: output of the machine learning pipeline, i.e. a trained classifier
3. app
    - run.py: Flask file to run the web application
    - templates contains html file for the web application

## 4. Requirements <a name="Requirements"></a>    

All of the requirements are captured in requirements.txt.  
```python
pip install -r requirements.txt
```

## 5. Instructions <a name="Instructions"></a>
1. Run the following commands in the project's root directory to process data and train model.

    - Run the ETL pipeline to clean data and store it in a SQLite database
		```python
		python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
		```      
    - Run the ML pipeline to trains and tunes a model using GridSearchCV, and save the model
		```python    
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
		```      
2. Run your web app with the following command in the app's directory.
		```python    
	    python run.py
		```          

3. Go to http://0.0.0.0:3001/

![alt text](https://github.com/binliu-base/dsnd-term2-p1-disaster-response/blob/master/screenshots/webpage.png)

