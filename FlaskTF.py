import flask
import os
import random
from flask import Flask, flash, request,jsonify, redirect
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import numpy as np
import skimage
import os
from skimage import io
import matplotlib.pyplot as plt
import random
import base64
import requests


def make_prediction(filename):

    global x,predict_tensor,sess
    tf.reset_default_graph() # why?
    initt = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initt)

    new_saver = tf.train.import_meta_graph('/Users/tunchiehhsu/Desktop/Image Classification/checkpoint_dir/MyModel.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('/Users/tunchiehhsu/Desktop/Image Classification/checkpoint_dir'))

    graph = tf.get_default_graph()
    x = graph.get_operation_by_name('xs').outputs[0]
    predict_tensor = tf.get_collection("pred_network")[0]

    img = io.imread(filename,as_gray = True)
    # reshape to uniform size
    reshape_img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    data = np.array(reshape_img.flatten())

    results = sess.run(predict_tensor,{x: data.reshape([1,65536])})
    confidence = results

    if confidence>0.5:
        prediction="undamaged"
    else:
        prediction="damaged"

    return prediction,str(confidence[0][0])

app = Flask(__name__)

UPLOAD_FOLDER = '/Users/tunchiehhsu/Desktop/Image Classification/upload'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg','png'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return "Index API"

@app.route('/classify', methods=['POST'])
def classify():
    prediction="None"
    confidence=0.
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename_input = secure_filename(file.filename)
            filename=os.path.join(app.config['UPLOAD_FOLDER'], filename_input)
            file.save(filename)
            print(filename)

            ###make a prediction from the file stored on computer
            prediction,confidence=make_prediction(filename)

            return_packet = {
                'prediction': prediction,
                'confidence': confidence
            }

            return jsonify(return_packet)
    return "Badly formed input"

@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    #init()
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True, port=8000)
