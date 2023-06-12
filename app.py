from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('executor1.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello hum 7 sath hai"


@app.route('/predict', methods=['POST'])
def predict():
    filetype = request.form.get('filetype')
    filesize = request.form.get('filesize')
    battery = request.form.get('battery')
    ram = request.form.get('ram')
    cpu = request.form.get('cpu')
    input_query = np.array([filetype, filesize, battery, ram, cpu])

    result = model.predict(input_query.reshape(1, -1))

    # Convert the result array to a regular Python list
    result_list = result.tolist()

    return jsonify({'total time': result_list})


if __name__ == '__main__':
    app.run(debug=True)
