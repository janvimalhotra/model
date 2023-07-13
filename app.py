from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
app = Flask(__name__,template_folder='templates')
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def hello():
    return render_template("p.html")
@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = []
    for value in request.form.values():
        try:
            int_features.append(float(value))
        except ValueError:
            # Handle the case when the value cannot be converted to a float
            return render_template('p.html', pred='Invalid input')

        final = np.array(int_features).reshape(1, -1)

    prediction=model.predict(final)
    if prediction==0:
        return render_template('p2.html')
    elif prediction==1:
        return render_template('p1.html')
    else:
        return render_template('p.html',pred='..')
if __name__=="__main__":
    app.run(host='localhost', port=5000)