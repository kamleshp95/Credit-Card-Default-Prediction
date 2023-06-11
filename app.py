from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

#Import pickle model file 
model=pickle.load(open("randomforestclassifiermodel.pkl","rb"))

# Home Page for the flask app
@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

# Predict function to gather all inputs and predict the outcome
@app.route('/predict', methods = ['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
   
    # re-arranging the list as per data set
    feature_list = [features[4]] + features[:4] + features[5:11][::-1] + features[11:17][::-1] + features[17:][::-1]
    features_arr = [np.array(feature_list)]


    # converting the array into a pandas dataframe
 
    col_names=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    df = pd.DataFrame(features_arr, columns= col_names)

    # doing the prediction
    prediction = model.predict(features_arr)

    result = ""
    if prediction == 1:
        result = "The credit card holder WILL BE DEFAULTER in the next month"
    else:
        result = "The Credit card holder WILL NOT BE DEFAULTER in the next month"
    
    # logging.info('Result returned !!')
    return render_template('index.html', prediction_text=result)


def predict():

    limit_bal = float(request.form['limit_bal'])
    sex = request.form['sex']
    age = int(request.form['age'])
    education = request.form['education']
    marriage = request.form['marriage']
    pay = request.form['payment_history']
    bill_amt = float(request.form['average_bill'])
    pay_amt1 = float(request.form['jan_payment'])
    pay_amt2 = float(request.form['feb_payment'])
    pay_amt3 = float(request.form['march_payment'])
    pay_amt4 = float(request.form['april_payment'])
    pay_amt5 = float(request.form['may_payment'])
    pay_amt6 = float(request.form['june_payment'])

# Module to convert categorical sex values to numerical values
    if sex == 'male':
        sex = 1
    else:
        sex = 2

# Education Module to convert categorical values into numerical
    education_dict = {
        'uneducated' : 0,
        'middle_school' : 1,
        'high_school' : 2,
        'ug' : 3,
        'pg' : 4,
        'phd' : 5,
        'others' : 6,
        }

    for name, value in education_dict.items():
        if education == name:
            education = value
            break

# Marriage module to assign the numrical value for categorical inputs
    marriage_dict = {
        'unmarried' : 0,
        'married' : 1,
        'divorced' : 2,
        'widowed' : 3
        }

    for name, value in marriage_dict.items():
        if marriage == name:
            marriage = value
            break

# Payment module to assign the numrical value for categorical inputs 
    pay_dict = {
        'on_time' : 0,
        'advance_one' : -1,
        'advance_two' : -2,
        'delay_one' : 1,
        'delay_two' : 2,
        'delay_three' : 3,
        'delay_four' : 4,
        'delay_five' : 5,
        'delay_six' : 6,
        'delay_seven' : 7
        }

    for name, value in pay_dict.items():
        if pay == name:
            pay = value
            break

# Module to predict the outcome using ML algorithm
    predict = model.predict([[limit_bal, sex, education, marriage, age
                , pay, bill_amt, pay_amt1, pay_amt2, pay_amt3, pay_amt4
                , pay_amt5, pay_amt6]])

# Converting the categorical values into intergers
    if predict == 0:
        output = 'Not Default'
    else:
        output = 'Default'

    return render_template('index.html', prediction_text = 'Credit Card Default Status for account {} : {}'.format( output))


# Driver Code
if __name__ == '__main__':
    app.run(debug = True)