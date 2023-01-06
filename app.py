import pickle
import joblib
import pandas as pd
import numpy as np
from flask import Flask,render_template,request

regressor = joblib.load('pslmodel_ridge.sav')
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html',val='')

@app.route('/predict',methods=['POST'])
def predict():
    a = []
    
    if request.method == 'POST':

        venue = request.form['venue']
        if venue=='Dubai International Cricket Stadium, Dubai':
            a = a + [1,0,0,0,0,0,0]

        elif venue=='Gaddafi Stadium, Lahore':
            a = a + [0,1,0,0,0,0,0]

        elif venue=='Multan Cricket Stadium, Multan':
            a = a + [0,0,1,0,0,0,0]

        elif venue=='National Stadium, Karachi':
            a = a + [0,0,0,1,0,0,0]

        elif venue=='Rawalpindi Cricket Stadium, Rawalpindi':
            a = a + [0,0,0,0,1,0,0]

        elif venue=='Sharjah Cricket Stadium, Sharjah':
            a = a + [0,0,0,0,0,1,0]

        elif venue=='Sheikh Zayed Stadium, Abu Dhabi':
            a = a + [0,0,0,0,0,0,1]            
  
        batting_team = request.form['batting-team']
        if batting_team == 'Islamabad United':
            a = a + [1,0,0,0,0,0]
        elif batting_team == 'Karachi Kings':
            a = a + [0,1,0,0,0,0]
        elif batting_team == 'Lahore Qalandars':
            a = a + [0,0,1,0,0,0]
        elif batting_team == 'Multan Sultans':
            a = a + [0,0,0,1,0,0]
        elif batting_team == 'Peshawar Zalmi':
            a = a + [0,0,0,0,1,0]
        elif batting_team == 'Quetta Gladiator':
            a = a + [0,0,0,0,0,1]
        

        bowling_team = request.form['bowling-team']
        if bowling_team == 'Islamabad United':
            a = a + [1,0,0,0,0,0]
        elif bowling_team == 'Karachi Kings':
            a = a + [0,1,0,0,0,0]
        elif bowling_team == 'Lahore Qalandars':
            a = a + [0,0,1,0,0,0]
        elif bowling_team == 'Multan Sultans':
            a = a + [0,0,0,1,0,0]
        elif bowling_team == 'Peshawar Zalmi':
            a = a + [0,0,0,0,1,0]
        elif bowling_team == 'Quetta Gladiator':
            a = a + [0,0,0,0,0,1]

        if batting_team==bowling_team and batting_team!='none' and bowling_team!='none':
            return render_template('home.html',val='Batting team and Bowling team cant be same and none of the values can not be empty.')
        

        overs = request.form['overs']
        runs = request.form['runs']
        wickets = request.form['wickets']
        runs_in_prev_5 = request.form['runs_in_prev_5']
        wickets_in_prev_5 = request.form['wickets_in_prev_5']

        if overs=='' or runs=='' or wickets=='' or runs_in_prev_5=='' or wickets_in_prev_5=='':
            return render_template('home.html',val='You can not leave any field empty')
        
        overs = float(overs)
        runs = int(runs)
        wickets = int(wickets)
        runs_in_prev_5 = int(runs_in_prev_5)
        wickets_in_prev_5 = int(wickets_in_prev_5)

        
        if overs < 5:
            return render_template('home.html', val= 'Over Should be 5 or greater then 5')

        a = np.array(a).reshape(1,-1)

        b = [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]
        b = np.array(b).reshape(1,-1)
        b = scaler.transform(b)

        data = np.concatenate((a,b),axis=1)


        my_prediction = int(regressor.predict(data)[0])

        return render_template('home.html', val=f'The final score will be around {my_prediction-5} to {my_prediction+10}.')

if __name__ == '__main__':
    app.run(debug=True)
