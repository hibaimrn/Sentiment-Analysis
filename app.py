from flask import Flask, render_template, request
import os
import pickle
from PIL import Image
import numpy as np
import statistics

app=Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER']=r'static\files'

def get_models():
    global dt
    global gnb
    global knn
    global log_reg
    global rf
    global sgd
    global vect
    pickle.load(open('dt.pkl', 'rb'))
    pickle.load(open('gnb.pkl', 'rb'))
    pickle.load(open('knn.pkl', 'rb'))
    pickle.load(open('log_reg.pkl', 'rb'))
    pickle.load(open('rf.pkl', 'rb'))
    pickle.load(open('sgd.pkl', 'rb'))
    pickle.load(open('vect.pkl', 'rb'))
    print('Models retrieved')

def model_predict(img_path):
    test_image=Image.open(img_path)
    test_image=test_image.resize((100,100))
    test_image=test_image.convert('L')
    test_image=np.array(test_image)
    result1=rf.predict(test_image)
    result2=log_reg.predict(test_image)
    result3=knn.predict(test_image)
    sen=["negative","neutral","positive", "very_negative", "very_positive"]
    results=[sen[result1[0]],sen[result2[0]],sen[result3[0]]]
    result=statistics.mode(results)
    return result

get_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(file_path)
        preds = model_predict(file_path)
        result=preds
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)