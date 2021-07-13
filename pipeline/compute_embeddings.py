import os
from pipeline.utils import DEFAULT_MODEL_NAME
from pipeline.utils import make_path_dicts
from pipeline.utils import detect_faces_alpha
from pipeline.utils import generate_embedding
from pipeline.utils import store_embeddings

import numpy as np
import cv2


class Embedding_DB(object):
    def __init__(self,
                 detection_model,
                 recognition_model,
                 database_path,
                 model_name: dict = DEFAULT_MODEL_NAME,
                 folder_path=None
                 ):
        self.database_path = database_path
        self.FD_model = detection_model
        self.FR_model = recognition_model
        self.fixed_height_for_recog = 720
        self.folder_path = folder_path
        self.model_name = model_name
        self.initialise_models()

    def initialise_models(self):
        '''
        First time execution of all models to reduce compute time
        '''
        # Initialise Detection Model
        blank_image = np.ones((512, 512, 3), np.uint8)
        _ = self.FD_model.detect(blank_image, 0.9)
        print("set Detection for Compute")
        # Initialise Recognition Model
        if self.model_name['Recognition'] == 'VGG-Face':
            blank_image_r = np.ones((1, 224, 224, 3), np.uint8)
        elif self.model_name['Recognition'] == 'ArcFace':
            blank_image_r = np.ones((1, 112, 112, 3), np.uint8)
        _ = self.FR_model.predict(blank_image_r)[0]
        print("set Recognition for Compute")

        if os.path.isfile(self.database_path):
            print("database path already exist and No new changes made")
        else:
            self.compute_all_embeddings()

    def compute_all_embeddings(self):
        import glob
        import os.path

        # Ignore, its just error checking
        if not os.path.isdir(self.folder_path):
            raise ValueError('Database Directory does not exist')

        if self.folder_path[-1] != '/':
            self.folder_path += '/'
        image_path = glob.glob(self.folder_path + '*/**')

        # Important part starts here
        known_Encodings = []
        known_Names = []

        person_2_path_dict = make_path_dicts(image_path)
        for key, value in person_2_path_dict.items():
            print(f"Encoding images of {value}")
            # detecting faces
            img = cv2.imread(key)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if (img.shape[2] == 4):
                img = img[:, :, :-1]
            else:
                img = img

            fixed_height = self.fixed_height_for_recog
            height_percent = (fixed_height / float(img.shape[0]))
            width_size = int((float(img.shape[1]) * float(height_percent)))

            faces = detect_faces_alpha(
                input=img,
                model=self.FD_model,
                align=True,
                width_size=width_size,
                fixed_height=fixed_height)

            # Generating Embeddings
            try:
                img1_embedding = generate_embedding(faces[0], self.FR_model,
                                                    'VGG-Face')
                known_Encodings.append(img1_embedding)
                known_Names.append(value)
            except: # noqa
                print(f"failed to encode {value}'s images")

        self.known_Encodings = known_Encodings
        self.known_Names = known_Names
        store_embeddings(known_Names, known_Encodings, self.database_path)
