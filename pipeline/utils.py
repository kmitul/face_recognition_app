############################################ Imports ###############################################################
from skimage.transform import resize
import cv2
import numpy as np
import math
from PIL import Image
import json
from collections import defaultdict

from deepface.commons import distance as dst
from tensorflow.python.ops.gen_math_ops import approximate_equal

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
############################################ Globals ###############################################################
DEFAULT_MODEL_NAME = {
    'Detection': 'RetinaFace',
    'Alignment_Detection': 'FAN',
    'Recognition': 'VGG-Face',  # 'ArcFace'
    'Mask': ''
}

mask_dict = {0: "No Mask", 1: "Mask", 2: "Mask Incorrect"}

thresholds = {
      'VGG-Face': {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75},
      'OpenFace': {'cosine': 0.10, 'euclidean': 0.55, 'euclidean_l2': 0.55},
      'Facenet':  {'cosine': 0.40, 'euclidean': 10, 'euclidean_l2': 0.80},
      'DeepFace': {'cosine': 0.23, 'euclidean': 64, 'euclidean_l2': 0.64},
      'DeepID':   {'cosine': 0.015, 'euclidean': 45, 'euclidean_l2': 0.17},
      'Dlib':     {'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.6},
      'ArcFace':  {'cosine': 0.6871912959056619,
                   'euclidean': 4.1591468986978075,
                   'euclidean_l2': 1.1315718048269017}
      }

thresholds_mask = {
      'VGG-Face': {'cosine': 0.30, 'euclidean': 0.45, 'euclidean_l2': 0.65},
      'OpenFace': {'cosine': 0.10, 'euclidean': 0.55, 'euclidean_l2': 0.55},
      'Facenet':  {'cosine': 0.40, 'euclidean': 10, 'euclidean_l2': 0.80},
      'DeepFace': {'cosine': 0.23, 'euclidean': 64, 'euclidean_l2': 0.64},
      'DeepID':   {'cosine': 0.015, 'euclidean': 45, 'euclidean_l2': 0.17},
      'Dlib':     {'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.6},
      'ArcFace':  {'cosine': 0.6871912959056619,
                   'euclidean': 4.1591468986978075,
                   'euclidean_l2': 1.1315718048269017}
      }
#################################################
#################################################
#  General Utils
#################################################
#################################################


def findThreshold(thresholds, metric, model):
    return thresholds[model][metric]


def make_path_dicts(paths: list):
    names = []
    person_2_path_dict = {}
    paths.sort()
    for each in paths:
        names.append(each.split('/')[-2])

    for i in range(len(names)):
        person_2_path_dict[paths[i]] = names[i]

    return person_2_path_dict


def compare_encodings(img1_embedding, img2_embedding,
                      metric, model_name, thresholds_set=thresholds):
    if metric == 'cosine':
        distance = dst.findCosineDistance(img1_embedding, img2_embedding)
    elif metric == 'euclidean':
        distance = dst.findEuclideanDistance(img1_embedding, img2_embedding)
    elif metric == 'euclidean_l2':
        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_embedding),
                                             dst.l2_normalize(img2_embedding))

    threshold = findThreshold(thresholds_set, metric, model_name)

    if distance <= threshold:
        return distance
    else:
        return -1

#################################################
#################################################
#  Used for Fan Network
#################################################
#################################################


def draw_bounding_box(frame, landmarks):
    h, w, c = frame.shape
    bb = []
    if landmarks:
        for handLMs in landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs:
                x, y = int(lm[0]), int(lm[1])
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            bb.append((x_min, y_min, x_max, y_max))
    return frame, bb


def bounding_box_crop_fan(img, bb: list):
    X = bb[0]
    Y = bb[1]
    W = bb[2] - bb[0]
    H = bb[3] - bb[1]
    cropped_image = img[Y:Y + H, X:X + W]
    return cropped_image


def detect_faces_fan(input, model):
    faces = []
    preds = model.get_landmarks(input)
    out, bb = draw_bounding_box(input, preds)
    for pred in bb:
        img_crop = bounding_box_crop_fan(out, pred)
        faces.append(img_crop)
    return faces


#################################################
#################################################
#  Used for Retina Face
#################################################
#################################################

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(
        np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment_procedure(img, left_eye, right_eye, nose):
    # this function aligns given face in img based on left and right eye
    # coordinates
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    upside_down = False
    if nose[1] < left_eye[1] or nose[1] < right_eye[1]:
        upside_down = True

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        if upside_down:
            angle = angle + 90

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    # -----------------------

    return img


def align_face(obj, img, align=True):

    resp = []

    if type(obj) == dict:
        for key in obj:
            identity = obj[key]

            facial_area = identity["facial_area"]
            facial_img = img[facial_area[1]:facial_area[3], facial_area[0]:
                             facial_area[2]]

            if align:
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                mouth_right = landmarks["mouth_right"] # noqa
                mouth_left = landmarks["mouth_left"] # noqa

                facial_img = alignment_procedure(facial_img, right_eye,
                                                 left_eye, nose)
            resp.append(facial_img)
    return resp


def detect_faces_alpha(input, model,
                       align=True, width_size=1920,
                       fixed_height=1080):

    if type(input) == str:
        input = cv2.imread(input)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

        # Handling corner case for 4 channels
        if (input.shape[2] == 4):
            input = input[:, :, :-1]
        else:
            input = input

        input = cv2.resize(input, (width_size, fixed_height))

    # Retina Face
    faces = model.detect(input, 0.9)
    faces = align_face(faces, input, align=align)

    return faces


#################################################
#################################################
#  Utils for calling FR models
#################################################
#################################################


def resizingimage(img, model_name):
    if (model_name == 'VGG-Face'):
        image_resized = resize(img, (224, 224), anti_aliasing=True)
    elif (model_name == 'ArcFace'):
        image_resized = resize(img, (112, 112), anti_aliasing=True)
    elif (model_name == 'Emotion'):
        image_resized = resize(img, (48, 48), anti_aliasing=True)
        out_image = image_resized.reshape(1, image_resized.shape[0],
                                      image_resized.shape[1],
                                      1)
        return out_image

    out_image = image_resized.reshape(1, image_resized.shape[0],
                                      image_resized.shape[1],
                                      image_resized.shape[2])
    return out_image


def findApparentAge(age_predictions):
	output_indexes = np.array([i for i in range(0, 101)])
	apparent_age = np.sum(age_predictions * output_indexes)
	return apparent_age

def generate_embedding(img1, model, model_name, age_model, emo_model, emotion_labels):
    emo_obj = {}

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = resizingimage(img1, model_name)
    img1_esize = resizingimage(img1_gray, 'Emotion')
    img1_embedding = model.predict(img1)[0]

    # Age and Emotion prediction 
    age_predictions = age_model.predict(img1)[0,:]
    apparent_age = round(findApparentAge(age_predictions))
    
    emotion_predictions = emo_model.predict(img1_esize)[0,:]
    sum_of_predictions = emotion_predictions.sum()

    emo_obj["emotion"] = {}

    for i in range(0, len(emotion_labels)):
        emotion_label = emotion_labels[i]
        emotion_prediction = round(100 * emotion_predictions[i] / sum_of_predictions)
        emo_obj["emotion"][emotion_label] = emotion_prediction
    
    emo_obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]

    return img1_embedding, apparent_age, emo_obj

#################################################
#################################################
#  Utils for calling Mask models
#################################################
#################################################

def resizingmtcnnfacemask(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (64,64))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return face


def store_embeddings(known_names, known_encodings, save_path):

    orDict = defaultdict(list)
    for name, enc in zip(known_names, known_encodings): # noqa

        if isinstance(enc, np.ndarray):
            enc = enc.tolist()

        orDict[name].append(enc)

    # print(orDict)

    with open(save_path, 'w') as fp:
        json.dump(dict(orDict), fp)


def get_embeddings(path):

    f = open(path, "r")
    file = json.load(f)
    known_names = []
    known_encodings = []
    for names in file:
        for embed in file[names]:
            known_names.append(names)
            known_encodings.append(np.array(embed))

    return known_names, known_encodings

###################################################################################################################