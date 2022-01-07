from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
@app.route('/')
def man():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    count_vect = pickle.load(open('./count_tfid.pickle','rb'))
    model = pickle.load(open('./my_model.pickle','rb'))
    pred = model.predict(count_vect.transform([data1]))
    return render_template('after.html', data=pred[0])
if __name__ == "__main__":
    app.run(debug=True)