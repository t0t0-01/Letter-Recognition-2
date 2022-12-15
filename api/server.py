from flask import Flask
from flask import request
import random
import io
import base64 
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from main import predict_letter_from_image
#from main import load_model
from pre_processing import get_bounding_box
import tensorflow as tf

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def load_cnn():
    new_model = tf.keras.models.load_model('../models/cnn_model')
    return new_model


def predict_letters_from_image(base64, model):
    bounded, letters = get_bounding_box("./temp", "letter.png")
    processed = []
    for letter in letters:
        temp = letter[np.newaxis, :, :, np.newaxis]
        prediction = model.predict(temp)
        prediction = classes[np.argmax(prediction)]
        processed.append(prediction)
    processed = processed[::-1]
    return bounded, "".join(list(processed))
    


model_english = load_cnn()
###########################
model_arabic = ""

app = Flask(__name__)


def analyze():
    nb = random.randint(0, 100)
    return str(nb)


def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    temp = Image.open(io.BytesIO(imgdata))
    final = cv2.cvtColor(np.array(temp), cv2.COLOR_BGR2RGB)
    return final
    


@app.route('/')
def main():
    return analyze()


@app.route("/upload-img", methods=["POST"])
def upload():
    if request.method == 'POST':
        string = request.json["base64"]
        language = request.json["language"]
        img_array = stringToImage(string)
        img = img_array[1450:2130, :, :]
        print(language)
        plt.imshow(img)
        plt.show()
        
        if np.mean(img) != 255:
            cv2.imwrite("./temp/letter.png", img)
            if language == 0:
                bounded, word = predict_letters_from_image(string, model_english)
                plt.imshow(bounded)
                plt.show()
            else:
                print("Arabic")
                #y = predict_letter_from_image(model_arabic)
            #return str(y[0])
            return word
        
        return ""

app.run("0.0.0.0", port=5000)