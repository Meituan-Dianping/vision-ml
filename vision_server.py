from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
from rcnn_predict import model_predict

app = Flask(__name__)
CORS(app)


@app.route('/client/vision', methods=["POST"])
def get_client_image():
    file = request.files['file']
    data = {
        "code": 0,
        "data": model_predict(file.read(), view=False)
    }
    return jsonify(data)


@app.errorhandler(Exception)
def error(e):
    ret = dict()
    ret["code"] = 1
    ret["data"] = repr(e)
    return jsonify(ret)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9092)
