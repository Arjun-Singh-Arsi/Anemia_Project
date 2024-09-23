from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('Anemia.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Predict', methods=['post'])
def anemia_predict():
    Sex = request.form.get('Sex')
    Red_Pixel = float(request.form.get('%Red Pixel'))
    Green_pixel = float(request.form.get('%Green pixel'))
    Blue_pixel = float(request.form.get('%Blue pixel'))
    Hb = float(request.form.get('Hb'))

    result = model.predict(np.array([Sex, Red_Pixel, Green_pixel, Blue_pixel, Hb]).reshape(1, 5))

    if result[0] == 1:
        result = 'Has Anemia'
    else:
        result = 'Has No Anemia'

    return render_template('index.html', result = result)


if __name__ == '__main__':
    app.run(debug=True)
