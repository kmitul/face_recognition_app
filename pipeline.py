from utils import *
from arcface import *
from retinaface import *

re_model = build_model()
arcmodel = loadModel()

# loading weights
Retinamodel = load_weights(re_model)
arcmodel.load_weights('models/arcface_weights.h5')

import time

t1 = time.time()
input_1 = io.imread('uploads/TomCruise/TC1.jpg')
input_2 = io.imread('uploads/TomCruise/TC2.jpg')

preds_1 = RetinaFace.detect_faces(input_1, model = Retinamodel)
preds_2 = RetinaFace.detect_faces(input_2, model = Retinamodel)

op1 = preds_1['face_1']
op2 = preds_2['face_1']

img_crop_1 = bounding_box_cropretina(input_1,op1)
img_crop_2 = bounding_box_cropretina(input_2,op2)

p1 = resizingarcface(img_crop_1)
p2 = resizingarcface(img_crop_2)

metrics = verify(p1, p2, 'cosine', model = arcmodel)
t2 = time.time()

print("time taken : ", t2 - t1)
