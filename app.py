from flask import Flask, render_template, request, url_for, send_from_directory, jsonify, json
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import time
import os

# Model Specific Imports
# from utils import *
# from arcface import *
# from retinaface import *


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/image'
app._static_folder = 'static'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == "POST":

        file1 = request.files['file1']
        file2 = request.files['file2']

        # Assumption made that we will have atleast 2 input images
        fn1 = secure_filename(file1.filename)
        fn2 = secure_filename(file2.filename)

        path1 = os.path.join(app.config['UPLOAD_FOLDER'], fn1)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], fn2)

        files[0].save(path1)
        files[1].save(path2)

        input_1 = io.imread(path1)
        input_2 = io.imread(path2)

        preds_1 = RetinaFace.detect_faces(input_1, model = Retinamodel)
        preds_2 = RetinaFace.detect_faces(input_2, model = Retinamodel)

        op1 = preds_1['face_1']
        op2 = preds_2['face_1']

        img_crop_1 = bounding_box_cropretina(input_1,op1)
        img_crop_2 = bounding_box_cropretina(input_2,op2)

        p1 = resizingarcface(img_crop_1)
        p2 = resizingarcface(img_crop_2)

        result = verify(p1, p2,'cosine', model=arcmodel)

        # SAVING THE PLOTTED IMAGE TO VISUALIZE IT
        fig = plt.figure()

        ax1 = fig.add_subplot(1,2,1)
        plt.axis('off')
        plt.imshow(img1[0])

        ax2 = fig.add_subplot(1,2,2)
        plt.axis('off')
        plt.imshow(img2[0])

        plt.show()

        # Some logic needed to uniquely name the prediction files
        plt.savefig('static/fig1.png')

        # REMOVING THE UPLOADED IMAGE AFTER THE WORK IS DONE
        os.remove(path1)
        os.remove(path2)

        return render_template('pred.html', img1 = fn1, img2 = fn2, result = result)

    if request.method == 'GET':
        return render_template('pred.html')


@app.route('/<path:filename>')
def send_file(filename):
    return send_from_directory('uploads/image', filename)


if __name__ == '__main__':

    # #loading the models and weights
    # re_model = build_model()
    # arcmodel = loadModel()
    #
    # Retinamodel = load_weights(re_model)
    # arcmodel.load_weights('models/arcface_weights.h5')

    app.run(debug=True)
