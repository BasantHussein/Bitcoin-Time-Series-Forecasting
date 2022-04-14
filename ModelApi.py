import json
from flask import Flask, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'predicts the bitcoin prices using dayofmonth, month and year.'})

@app.route('/predicts_price/<dayofmonth>/<month>/<year>', methods=["GET"])
def predicts_price(dayofmonth,month,year):
    model = pickle.load(open('RidgeModel.pkl', 'rb'))
    data = {'dayofmonth':[dayofmonth],
            'month': [month],
            'year': [year]}

    df_test = pd.DataFrame(data)
    lists = model.predict(df_test).tolist()

    return  json.dumps(np.exp(lists[0]))



if __name__ == '__main__':
    app.run()