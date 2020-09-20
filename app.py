import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
app=Flask(__name__)
model=pickle.load(open('GaussianNB.pkl','rb'))

@app.route('/')
def home():
    return render_template('final.html')

@app.route('/stress_detection',methods=['POST','GET'])
def stress_detection():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]    
    print(x_test)
    sc = load('StandardScalar.save') 
    prediction = model.predict(sc.transform(x_test))
    print(prediction)
    output=prediction[0]
    if(output==0):
        pred="Your stress level is normal.No need of taking treatment its fine." 
    else:
        pred="Your stress level is above average and it is reaching to high...Take treatment as soon as possible before facing any danger situations"
    
    return render_template('final.html', prediction_text='{}'.format(pred))

@app.route('/predict_api',methods=['POST']) 
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.stress_detecion([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)