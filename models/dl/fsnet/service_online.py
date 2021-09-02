__author__ = 'dk'
from flask import request, Flask, jsonify
app = Flask(__name__)
from fsnet_main_model import model

fsnet_model = model('fgnet53', randseed= 128, splitrate=0.1)
@app.route('/fsnet/logit',methods=['POST'])
def get_logit():
    if not request.json or not 'flow' in request.json:
        return jsonify({'error':'not flow in request'})
    flow = request.json['flow']
    logit =fsnet_model.logit_online(flow).tolist()
    return  jsonify({'logit':logit})

if __name__ == '__main__':
    app.run(host='192.168.255.82',port=10086,debug=True)