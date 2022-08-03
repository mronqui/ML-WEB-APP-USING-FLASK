import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='')

model = pickle.load(open('../models/titanic_model.pickle', 'rb'))

@app.route('/') #http://www.google.com/
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Would you survive? {} (1=survived, 0=deceased)'.format(output))

if __name__=="__main__":
    app.run(port=5000, debug=True)

