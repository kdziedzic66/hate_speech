import os
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request

from pipeline import Pipeline
from utils.constants import PRODUCTION_MODEL_NAME
from utils.helpers import download_model_from_s3

download_model_from_s3(model_name=PRODUCTION_MODEL_NAME)
PIPELINE = Pipeline.from_model_id(PRODUCTION_MODEL_NAME)
EXECUTOR = ThreadPoolExecutor(max_workers=1)


app = Flask(__name__)


@app.route("/hate_speech_detection", methods=["GET"])
def hate_speech_detection():
    request_data = request.json
    text = request_data["text"]
    prediction = EXECUTOR.submit(PIPELINE.predict, text).result()
    return jsonify(**prediction)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        debug=True,
        threaded=True,
        use_reloader=False,
        port=os.environ.get("PORT", "5000"),
    )
