from RetinaFacetf2.src.retinafacetf2.retinaface import RetinaFace
from deepface.basemodels import VGGFace
from keras.models import load_model
from FR_engine import FR_Engine
from compute_embeddings import Embedding_DB
import cv2
from models import loadModel_emotion, loadModel_age, loadModel_mask

Predict = True


if Predict:
    """
    To Directly call models and verify based the computed embeddings
    directly stored in json path.

    - Initilize all the models and choose the json path
    - pass the frame in process_frame()

    """
    # Detector Backend 
    detector1 = RetinaFace(False, 0.4)

    # Face Recognition Models 
    VGGFace_model = VGGFace.loadModel()
    VGGFace_model.load_weights('./weights/vgg_face_weights.h5')

    # Utility models 
    agemodel = loadModel_age()
    emomodel = loadModel_emotion()
    fmmodel = loadModel_mask()
    
    print(f'Models Loaded!')

    # Database Path 
    json_path = "./db/real_madrid.json"

    engine = FR_Engine(detector1,
                       VGGFace_model,
                       fmmodel,
                       agemodel,
                       emomodel,
                       saved_embeddings_path=json_path)

    ##################################
    # Example 1
    ##################################
    frame_path = "/home/ubuntu/data/datasets/Real Madrid/RealMadrid.jpg"
    img = cv2.imread(frame_path)
    answer = engine.process_frame(img)
    cv2.imwrite("outputs/test1.jpg", answer["frame"])

    ##################################
    # Example 2
    ##################################

    frame_path = "/home/ubuntu/data/datasets/Real Madrid/Real-Madrid-7.jpg"
    img = cv2.imread(frame_path)
    answer = engine.process_frame(img)
    cv2.imwrite("outputs/test2.jpg", answer["frame"])

else:
    detector1 = RetinaFace(False, 0.4)
    VGGFace_model = VGGFace.loadModel()
    VGGFace_model.load_weights('/home/ubuntu/pipeline/weights/vgg_face_weights.h5')
    database_path = "/home/ubuntu/data/datasets/Real Madrid/"

    engine = Embedding_DB(detector1,
                          VGGFace_model,
                          "real_madrid.json",
                          folder_path=database_path)
