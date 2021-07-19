# flask 관련 패키지
from flask import Flask
from flask import request
from flask_cors import CORS

# 그외 패키지
import numpy as np
from keras.models import load_model
from PIL import Image
import logging


app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def hello():
    app.logger.info("router / IN")
    return {"hello": "world"}


# url/mnist/predict
@app.route("/mnist/predict", methods=['POST'])
def predict():
    app.logger.info("route /mnist/predict IN")

    requestData = Image.open(request.files['file']).resize((28, 28)).convert("L")

    img = np.resize(requestData, (1, 784))
    img = ((np.array(img) / 255) - 1) * -1

    return {"result": mnist_model.predict(img).tolist()}


if __name__ == '__main__':
    app.logger.info("__name__ == __main__ ")

    mnist_model = load_model('mnist_model.h5')
    app.logger.info("mnist model")

    app.run(host="0.0.0.0")
