from flask import Flask, render_template, request
import pickle

filename = 'voting_clf_DATA300.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('countvector_300DATA.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if message == '':
            return render_template('index.html', valid=0)
        else:
            data = [message]
            vect = cv.transform(data).toarray()
            my_prediction = classifier.predict(vect)
            return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)