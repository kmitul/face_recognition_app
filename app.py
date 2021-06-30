from flask import Flask, render_template, request, url_for, send_from_directory, jsonify, json, Response, session
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import time
import os

import matplotlib.pyplot as plt
from skimage import io

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads/image')
app._static_folder = os.path.join(os.getcwd(),'static')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare', methods=['GET','POST'])
def upload():
    if request.method == "POST":

        file1 = request.files['file1']
        file2 = request.files['file2']

        # Assumption made that we will have atleast 2 input images
        fn1 = secure_filename(file1.filename)
        fn2 = secure_filename(file2.filename)

        path1 = os.path.join(app.config['UPLOAD_FOLDER'], fn1)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], fn2)

        file1.save(path1)
        file2.save(path2)

        input_1 = io.imread(path1)
        input_2 = io.imread(path2)

        # preds_1 = RetinaFace.detect_faces(input_1, model = Retinamodel)
        # preds_2 = RetinaFace.detect_faces(input_2, model = Retinamodel)
        #
        # op1 = preds_1['face_1']
        # op2 = preds_2['face_1']
        #
        # img_crop_1 = bounding_box_cropretina(input_1,op1)
        # img_crop_2 = bounding_box_cropretina(input_2,op2)
        #
        # p1 = resizingarcface(img_crop_1)
        # p2 = resizingarcface(img_crop_2)
        #
        # result = verify(p1, p2,'cosine', model=arcmodel)
        result = {
            "verification": True,
            "score": 10
        }

        # SAVING THE PLOTTED IMAGE TO VISUALIZE IT
        fig = plt.figure()

        ax1 = fig.add_subplot(1,2,1)
        plt.axis('off')
        plt.imshow(input_1)

        ax2 = fig.add_subplot(1,2,2)
        plt.axis('off')
        plt.imshow(input_2)

        # plt.show()

        # Some trivial logic to uniquely name the prediction files
        plotname = 'result'+ fn1.split('.')[0] + fn2.split('.')[0]+'.png'
        plt.savefig(app._static_folder +'/plots/'+ plotname)

        # REMOVING THE UPLOADED IMAGE AFTER THE WORK IS DONE
        os.remove(path1)
        os.remove(path2)

        return render_template('pred.html', plot = plotname, result = result)

    if request.method == 'GET':
        return render_template('compare.html')

######################################################################################
from feed import WebcamFeed

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_stream')
def video_stream():
    return Response(gen(WebcamFeed()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/feed')
def feed():
    return render_template('stream.html')
#######################################################################################

@app.route('/static/plots/<path:filename>')
def send_plot(filename):
    return send_from_directory(app._static_folder + '/plots', filename)


@app.route('/static/bg/<path:filename>')
def send_bg(filename):
    return send_from_directory(app._static_folder + '/bg', filename)


if __name__ == '__main__':

    # #loading the models and weights
    # re_model = build_model()
    # arcmodel = loadModel()
    #
    # Retinamodel = load_weights(re_model)
    # arcmodel.load_weights('models/arcface_weights.h5')

    app.run(debug=True, host='0.0.0.0', port=80)
