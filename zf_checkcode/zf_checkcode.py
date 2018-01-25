from flask import Flask, request, jsonify
import base64
import tensorflow as tf
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import io, re
from PIL import Image

index = [0] * 33
# index_tmp = {'8': 8, 'v': 29, '2': 2, '7': 7, 'w': 30, 'j': 18, '5': 5, 'p': 23, 'm': 21, 't': 27, '4': 4, '6': 6, 's': 26, 'b': 10, '1': 1, 'k': 19, 'd': 12, 'f': 14, 'g': 15, 'x': 31, 'r': 25, 'i': 17, '0': 0, '3': 3, 'h': 16, 'c': 11, 'n': 22, 'e': 13, 'u': 28, 'l': 20, 'q': 24, 'a': 9,'y': 32}
index_tmp = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 'a': 9, 'b': 10, 'c': 11, 'd': 12,
             'e': 13, 'f': 14, 'g': 15, 'h': 16, 'i': 17, 'j': 18, 'k': 19, 'l': 20, 'm': 21, 'n': 22, 'p': 23, 'q': 24,
             'r': 25, 's': 26, 't': 27, 'u': 28, 'v': 29, 'w': 30, 'x': 31, 'y': 32}
for k, v in index_tmp.items():
    index[v] = k

app = Flask(__name__)


def init():
    import os
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'final.h5')
    model = load_model(path)
    graph = tf.get_default_graph()
    return model, graph


def depoint(img):  # input: gray image
    pixdata = img.load()
    w, h = img.size
    for i in [0, h - 1]:
        for j in range(w):
            pixdata[j, i] = 255
    for i in [0, w - 1]:
        for j in range(h):
            pixdata[i, j] = 255
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            count = 0
            if pixdata[x, y - 1] > 245:
                count = count + 1
            if pixdata[x, y + 1] > 245:
                count = count + 1
            if pixdata[x - 1, y] > 245:
                count = count + 1
            if pixdata[x + 1, y] > 245:
                count = count + 1
            if count > 2:
                pixdata[x, y] = 255
    return img


def handle_image(content):
    pic = io.BytesIO(content)
    pic = Image.open(pic).convert('1')
    pic = depoint(pic)
    y_min, y_max = 0, 22
    split_lines = [5, 17, 29, 41, 53]
    images = [pic.crop([u, y_min, v, y_max])
              for u, v in zip(split_lines[:-1], split_lines[1:])]
    result = ''

    model, graph = init()
    with graph.as_default():
        for i in images:
            bitio = io.BytesIO()
            i.save(bitio, 'png')
            bitio.seek(0)
            image = load_img(bitio, grayscale=True, target_size=(12, 22))
            image = img_to_array(image)
            image = np.resize(image, (1, 12, 22, 1))
            image /= 255
            y_prob = model.predict(image)
            y_classes = y_prob.argmax(axis=-1)
            result += index[y_classes[0]]
    return result


@app.route('/', methods=['POST'])
def checkcode_detect():
    # try:
    content = base64.b64decode(request.values.get('image', None))
    return jsonify(status='ok', result=handle_image(content))
    # except:
    #     return jsonify(status='no', result=None)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1928, debug=True)
