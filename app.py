##################### IMPORTING ALL REQUIRED PACKAGES FOR THE APPLICATION #############################################
import os
import string    
import random   
import json
import time
import glob
import base64

import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image

from flask import Flask, render_template, request, url_for, send_from_directory, session
from werkzeug.utils import redirect, secure_filename

import face_alignment
from pipeline.RetinaFacetf2.src.retinafacetf2.retinaface import RetinaFace
from deepface.basemodels import VGGFace
from keras.models import load_model
from pipeline.FR_engine import FR_Engine
from pipeline.compute_embeddings import Embedding_DB
from pipeline.models import loadModel_emotion, loadModel_age, loadModel_mask
from sheet_api.auth import add_info
################################### APP CONFIGURATION AND BASIC UTILITY ROUTES #######################################

# Initializing the flask app
app = Flask(__name__)

# Setting configurations of serving static files
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads/image')
app._static_folder = os.path.join(os.getcwd(),'static')

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for team information
@app.route('/team')
def team():
    return render_template('team.html')

# Route for sending static background files
@app.route('/static/bg/<path:filename>')
def send_bg(filename):
    return send_from_directory(app._static_folder + '/bg', filename)


############################################### VERIFICATION FEATURE ###########################################################
# Route resposible for the whole verification feature
@app.route('/compare', methods=['GET','POST'])
def upload():
    if request.method == "POST":

        # Receiving two images
        file1 = request.files['file1']
        file2 = request.files['file2']

        # Saving with secure filenames which removes spaces from filenames with underscores
        fn1 = secure_filename(file1.filename)
        fn2 = secure_filename(file2.filename)

        # Handling the case of same uploaded images
        if fn1 == fn2:
            fn2 = fn1.split('.')[0] + fn2

        # Saving files to upload folder
        path1 = os.path.join(app.config['UPLOAD_FOLDER'], fn1)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], fn2)

        file1.save(path1)
        file2.save(path2)

        # Sending the files as input to the FR Engine for detection process
        input_1 = io.imread(path1)
        input_2 = io.imread(path2)
    
        faces1, aligned_faces1, org_img_1 = engine.detection_process(input_1, verification_step=True)
        faces2, aligned_faces2, org_img_2 = engine.detection_process(input_2, verification_step=True)

        # Check when no faces are detected
        if type(faces1) == tuple or type(faces2) == tuple:
            
            result = {
                        "verification": -1,
                        "threshold": -1,
                        "distance": -1
                    }
            os.remove(path1)
            os.remove(path2)
            return render_template('pred.html', plot="-1", result=result)

        
        # Adding Visuals
        output_1=engine.add_visuals_verify(org_img_1, faces1)
        output_2=engine.add_visuals_verify(org_img_2, faces2)

        result=engine.verify(aligned_faces1[0], aligned_faces2[0], 'cosine')
        
        # CREATING PLOTS TO VISUALIZE THE PREDICTIONS
        fig = plt.figure()

        ax1 = fig.add_subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(output_1)

        ax2 = fig.add_subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(output_2)

        # Some trivial logic to uniquely name the prediction files
        plotname = 'result' + fn1.split('.')[0] + fn2.split('.')[0]+'.png'
        plt.savefig(app._static_folder + '/plots/' + plotname)

        # REMOVING THE UPLOADED IMAGE AFTER THE WORK IS DONE
        os.remove(path1)
        os.remove(path2)

        # Sending the result back to the client
        return render_template('pred.html', plot=plotname, result=result)
    

    
    if request.method == 'GET':
        # No more saving the plots on server
        # Removing them as soon as user wants to try different images
        filelist = glob.glob(os.path.join(app._static_folder, "plots", "*.png"))
        for f in filelist:
          os.remove(f)
        return render_template('compare.html')

# Route  for sending the results of verification feature
@app.route('/static/plots/<path:filename>')
def send_plot(filename):
    return send_from_directory(app._static_folder + '/plots', filename)


############################################### RECOGNITION FEATURE ####################################################
def age_group(age):
    if age <= 12:
        return "Child(0-12)"
    elif age <= 19:
        return "Teenager(13-19)"
    elif age <= 28:
        return "Youth(20-28)"
    elif age <= 45:
        return "Adult(29-45)"
    elif age <= 65:
        return "Middle Aged(46-45)"
    else:
        return "Senior Citizen(65+)"

@app.route('/recog', methods=['GET','POST'])
def recog():
    if request.method == "POST":

        # Getting file from the client and saving to server
        file1 = request.files['file1']
        fn1 = secure_filename(file1.filename)
        path1 = os.path.join(app.config['UPLOAD_FOLDER'], fn1)
        file1.save(path1)

        # Sending it to FR Engine for further recognition with the users embeddings in the database
        input_1 = cv2.imread(path1)
        answer = engine.process_frame(input_1, verification_step = True)

        # Checking the result of mask prediction
        for i in range(len(answer['mask'])):
            if(answer['mask'][i] == 1):
                answer['mask'][i] = "Mask"
            else : 
                answer['mask'][i] = "No Mask"

        # Setting age group based on the user's age
        answer['age'][0] = age_group(answer['age'][0])
        
        # Creating plot to visualze the predictons
        fig = plt.figure()

        ax1 = fig.add_subplot(1,2,1)
        plt.axis('off')
        plt.imshow(answer['frame'])

        # Some trivial logic to uniquely name the prediction files
        plotname = 'result'+ fn1.split('.')[0] + '.jpg'
        cv2.imwrite(os.path.join(app._static_folder,"plots",plotname), answer["frame"])

        # REMOVING THE UPLOADED IMAGE AFTER THE WORK IS DONE
        os.remove(path1)

        return render_template('pred_static.html', plot=plotname, result=answer)
    
    if request.method == 'GET':
        # No more saving the plots on server
        # Removing them as soon as user wants to try different images
        filelist = glob.glob(os.path.join(app._static_folder, "plots", "*.png"))
        for f in filelist:
          os.remove(f)
        return render_template('upload_recog.html')


##################################### ATTENDANCE MARKING APP ###########################################################


# Route for serving the static Javascript files
@app.route("/<path:filename>")
def send_file(filename):
    return send_from_directory(app._static_folder, filename)

# Route for sending the server side result back to the client
@app.route("/static/results/<path:filename>")
def send_result(filename):
    return send_from_directory(os.path.join(app._static_folder,"results"), filename)

# Route responsible for the whole Mark Attendance Feature
@app.route('/capture', methods = ['GET','POST'])
def capture_pred():
    if request.method == 'POST':

        # Receiving the images from client and saving it to server
        image_data = request.form.get("content").split(",")[1]
        with open("static/client.jpg", 'wb') as f:
            f.write(base64.b64decode(image_data))

        # Sending the file to Face Recognition Engine for various predictions
        image = cv2.imread(os.path.join(app._static_folder,"client.jpg"))
        answer = engine.process_frame(image)

        # Logic for giving unique names to the predicted images
        S = 10  # number of characters in the string.  
        ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = S))   
        plotname = 'result'+ str(ran) + '.jpg'

        # Saving the image which would be sent back to the client using above utility routes
        cv2.imwrite(os.path.join(app._static_folder,"results",plotname), answer["frame"])

        # Setting the age-group
        print(answer['age'])
        print(age_group(answer['age'][0]))
        
        answer['age'][0] = age_group(answer['age'][0])


        # Calling Sheet API for storing the records of recognized users
        add_info(answer)

        # Checking whether the person was wearing a mask or not
        for i in range(len(answer['mask'])):
            if(answer['mask'][i] == 1):
                answer['mask'][i] = "Mask"
            else : 
                answer['mask'][i] = "No Mask"

        # Rendering the response back to the frontend
        return render_template('pred_recognize.html', plot=plotname, result=answer)
    
    else:
        # Capturing the image in case of GET request
        return render_template('capture.html')

# Feature to list down all the registered users in the database
@app.route('/users')
def user_list():
    # Reading users from our database
    f = open('./sheet_api/employee_info.json', 'r')
    db = json.load(f)
    i = 1

    # Preparing entries in frontend-friendly format
    users = []
    for name in db.keys():
        users.append([db[name]['ID'], name, db[name]['Position'], db[name]['Department']])
    # print(users)
    return render_template('users.html', users = users)


#############################################################################################################################

if __name__ == '__main__':

    # Detector Backend
    detector1 = RetinaFace(False, 0.4)

    # Face Recognition Models 
    VGGFace_model = VGGFace.loadModel()
    VGGFace_model.load_weights('./pipeline/weights/vgg_face_weights.h5')

    # Utility models 
    agemodel = loadModel_age()
    emomodel = loadModel_emotion()
    fmmodel = loadModel_mask()
    
    print(f'Models Loaded!')

    # Database Path 
    json_path = "./db/OtsukaDB.json"

    # Initializing the FR engine
    engine = FR_Engine(detector1,
                       VGGFace_model,
                       fmmodel,
                       agemodel,
                       emomodel,
                       saved_embeddings_path=json_path)
    
    # Enabled SSL context for allowing HTTPS(443) traffic in our application 
    app.run(debug=False, host='0.0.0.0', port=443, ssl_context = 'adhoc')

