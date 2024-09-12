from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from googletrans import Translator
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

colunas = ['tamanho', 'ano', 'garagem']
modelo = pickle.load(open('../../models/modelo.sav','rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'julio'
app.config['BASIC_AUTH_PASSWORD'] = 'alura'

#app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
#app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return 'Minha Primeira API'

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    translator = Translator()
    frase_en = translator.translate(frase, dest='en')
    polaridade = TextBlob(frase_en.text).sentiment.polarity
    return 'polaridade {}'.format(polaridade)
    
@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():    
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])
    
app.run(debug=True, host='0.0.0.0')