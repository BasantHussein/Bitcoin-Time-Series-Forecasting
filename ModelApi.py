import json
from flask import Flask, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'predicts the bitcoin prices using month and year.'})

@app.route('/predicts_price/<month>/<year>', methods=["GET"])
def predicts_price(month,year):
    model = pickle.load(open('model.pkl', 'rb'))
    data = {'month': [month],
            'year': [year]}

    df_test = pd.DataFrame(data)
    lists = model.predict(df_test).tolist()

    return  json.dumps(np.exp(lists[0]))



if __name__ == '__main__':
    app.run()