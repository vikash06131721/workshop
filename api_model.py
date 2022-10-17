# -*- coding: UTF-8 -*-

from flask import request, jsonify, Response
from model import return_predictions
from json_logger import get_logger
logger = get_logger()
from flask import Flask
import json
from flair.models import SequenceTagger
import flair,torch
flair.device = torch.device('cpu')
model = SequenceTagger.load('model/resources/taggers/example-ner/best-model.pt')

logger.info('Creating Flask application context')
app = Flask(__name__, static_folder='static', template_folder='templates')  # type: Flask # Initialize the Flask application
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device="cpu"
model.to(device)


@app.route('/ner_engine', methods=['POST'])
def process_ner():
    global model
    content = json.loads(request.data)
    ans, status = return_predictions(content["text"],model)

    ans_obj = {
            'modified_string': ans,
            'status': status
        }
        
    response_ = jsonify(ans)
    response_.status_code = status
    
    return response_

@app.route('/healthcheck', methods=['GET'])
def health_check():
    resp = Response()
    resp.data = json.dumps({"status":"All Good!"})
    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8005, threaded=True)
