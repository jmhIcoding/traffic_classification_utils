import flask
from df_main_model import model
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
df_model = model('datacon_training',128,0.1)

_labels = ['0.json','1.json','2.json','3.json','4.json','5.json','6.json','7.json','8.json','9.json','10.json']
_labels.sort()

@app.route(rule= '/datacon', methods=['POST'])
def tunnel():
    try:
        request_data = request.json
        label = df_model.predict(request_data['packet_length'])
        label = [_labels[_id].replace('.json','') for _id in label]
        return jsonify({'status':'success', 'label': label})

    except BaseException as exp:
        #raise exp
        return  jsonify({'status':'error', 'data': str(exp)})

if __name__ == '__main__':
    app.run(host="0.0.0.0",
            port=8898,
            threaded=True)
