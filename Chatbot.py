from colorama import Fore, Style
from flask import Flask, request, jsonify
from keras.models import load_model
from tensorflow import keras
import numpy as np
import colorama
import pickle
import json
import os
colorama.init()
app = Flask(__name__)

code_path = "C:\\Users\\user\\Python_Anaconda\\task\\06\\Code"
data_path = "C:\\Users\\user\\Python_Anaconda\\task\\06\\DataSet"
results_path = "C:\\Users\\user\\Python_Anaconda\\task\\06\\ChatbotResults"

json_file_path = os.path.join(data_path,'ChatbotData.json')
model_info_path = os.path.join(results_path,'model info')
images_path = os.path.join(results_path, "images")

def load_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def save_object(obj, filename,path):
    filename = os.path.join(path,filename)
    with open(filename+".pkl", 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()


def load_object(filename,path):
    filename = os.path.join(path,filename)
    with open(filename+".pkl", 'rb') as outp:
        loaded_object = pickle.load(outp)
    outp.close()
    return loaded_object

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data_ = load_json_file(json_file_path)
        model_name = 'chat_model'
        model = load_model(filepath=os.path.join(model_info_path ,model_name+".h5"))
        tokenizer = load_object("tokenizer",results_path)
        lbl_encoder = load_object("label_encoder",results_path)
        max_len = 200
        data = request.get_json()
        question = data.get('question')
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(question),
                                                                          truncating='post', maxlen=max_len),verbose=0)
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        # answer = {"answer": ""}
        for i in data_['intents']:
            if i['tag'] == tag:
                answer = ["ChatBot: ", np.random.choice(i['responses'])]
        return jsonify({"answer" : answer})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)