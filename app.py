from flask import Flask
from flask import jsonify
import os
from esepteKomnata import EsepteKomnata

app = Flask(__name__)

# EsepteKomnata should loaded one time, because it's too long proccess
esepteKomnata = EsepteKomnata()

@app.route("/")
def route0():
    return "Esepte API"

@app.route("/komnata/<imageName>")
def route02(imageName):

    answer = esepteKomnata.getKomnataType(imageName)

    answer_dict = { "komnataType": '' }

    if answer is not None:
        answer_dict = { "komnataType": answer }

    return jsonify(answer_dict)

# export FLASK_APP=esepteAIServer.py 
# flask run --port=6012

# set FLASK_APP=app.py 
# python3 -m flask run --port=6012