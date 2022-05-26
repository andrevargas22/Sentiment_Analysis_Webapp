# Sentiment Analysis Webapp
Small web app that runs a torch trained model for sentiment analysis based on movie reviews

### Step 1 - Clone this repository

### Step 2 - Install Requirements:

```
> pip install -r requirements.txt
```

### Step 3 - Run train.py to generate the model:
```
> python train/train.py
```

### Step 4 - Set the app as FLASK_APP

On cmd:
```
> set FLASK_APP=app/sentiment
```
On PowerShell:
```
> $env:FLASK_APP="app/sentiment.py"
```
On Linux:
```
> export FLASK_APP=app/sentiment
``` 

### Step 5 - Run server
```
> flask run
```

### Results:

![alt text](https://github.com/andrevargas22/Sentiment_Analysis_Webapp/blob/main/img/1.png)

![alt text](https://github.com/andrevargas22/Sentiment_Analysis_Webapp/blob/main/img/2.png)

