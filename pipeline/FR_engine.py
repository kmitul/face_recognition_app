############################################ Imports ###############################################################
from pipeline.utils import DEFAULT_MODEL_NAME, mask_dict, thresholds, thresholds_mask
from pipeline.utils import make_path_dicts, compare_encodings
from pipeline.utils import align_face, detect_faces_alpha
from pipeline.utils import generate_embedding
from pipeline.utils import resizingmtcnnfacemask, get_embeddings
from pipeline.utils import resizingimage,findThreshold
from deepface.commons import functions, distance as dst
import numpy as np
import cv2
import time
############################################ Class definition ###############################################################
class FR_Engine(object):
    '''
    Pipeline
    1) Detection - [RetinaFace]
      1.1) Single Face - exception - [FaceAlignment]

    2) Crop & Align
    3)
      a) Face Recognition - [VGG Face, ArcFace]
      b) Mask Recognition - [Mask Model]

    Rest API -

    input - frame

    output -

    1. bounding boxes, landmarks (coming detection module)
    2. processed frame with bounding boxes (detection module)
    3. Every bounding box ka, identity (Recognition module)
    4. Every bounding box ka, mask detection (Mask Module)

    RetinaFace - detection
    Haardcascade OpenCV
    '''

    def __init__(
            self,
            detection_model1,
            recognition_model,
            mask_model=None,
            age_model=None,
            emo_model=None,
            database_path=None,  # FOLDER
            saved_embeddings_path=None,
            model_name: dict = DEFAULT_MODEL_NAME,
            alignment_model=None,
            **kwargs):
        self.model_name = model_name
        self.width_size = 640
        self.fixed_height = 480
        self.fixed_height_for_recog = 720

        # Models
        self.FD_model = detection_model1
        self.FR_model = recognition_model
        self.Mask_model = mask_model
        self.age_model = age_model
        self.emo_model = emo_model

        # Embeddings
        self.saved_embeddings_path = saved_embeddings_path  # json_path
        self.database_path = database_path

        # Miscelleneous 
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.initialise_models()

    def initialise_models(self):
        '''
        First time execution of all models to reduce compute time
        '''
        t1 = time.time()
        # Initialise Detection Model
        blank_image = np.ones((512, 512, 3), np.uint8)
        blank_image = cv2.resize(blank_image, (self.width_size,
                                               self.fixed_height))
        _ = self.FD_model.detect(blank_image, 0.9)
        print("set Detection")
        # Initialise Recognition Model
        if self.model_name['Recognition'] == 'VGG-Face':
            blank_image_r = np.ones((1, 224, 224, 3), np.uint8)
        elif self.model_name['Recognition'] == 'ArcFace':
            blank_image_r = np.ones((1, 112, 112, 3), np.uint8)
        _ = self.FR_model.predict(blank_image_r)[0]
        print("set Recognition")

        # Initialize Util Models
        blank_image_e = np.ones((1, 48, 48, 1), np.uint8)

        _ = self.age_model.predict(blank_image_r)[0,:]
        _ = self.emo_model.predict(blank_image_e)[0,:]
        print("set emo and age!")
        # Initialise Mask Model

        # Initialise FAN Alignment Model
        blank_image = np.ones((512, 512, 3), np.uint8)
        # preds = self.alignment_model.get_landmarks(blank_image)

        blank_image = resizingmtcnnfacemask(blank_image)
        _ = self.Mask_model.predict(blank_image)
            
        known_names, known_encodings = get_embeddings(self.saved_embeddings_path)
        self.known_Encodings = known_encodings
        self.known_Names = known_names
        print("Loaded Embeddings Directly")
        print(f"Initialistion done in {time.time() - t1}s")

    def process_frame(self, frame, verification_step = False):
        '''
        process the full frame here
        '''

        t1 = time.time()
        boxes, aligned_faces, org_img = self.detection_process(frame, verification_step = verification_step)  # 0.4
        print(f"Detection done in {time.time() - t1}s")
        t1 = time.time()
        bounding_boxes, identities, mask_status, ages, emotions = self.recognition_process(boxes, aligned_faces)  # 0.1
        print(f"Recognition & Mask done in {time.time() - t1}s")
        visuals = self.add_visuals(org_img,
                                   bounding_boxes,
                                   identities,
                                   mask_status,
                                   ages,
                                   emotions)  # 0.05

        return {
            "frame": visuals,  # io.bytes()
            "bounding_boxes": bounding_boxes,
            "employee": identities,
            "mask": mask_status,
            "age": ages,
            "emotions" : emotions
        }

    def detection_process(self, frame, align=True, verification_step = False):
        '''
        No visulisations, just return a dictionary & cropped images
        '''
        if type(frame) is str:
            frame = cv2.imread(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Handling corner case for 4 channels
        if (frame.shape[2] == 4):
            frame = frame[:, :, :-1]
        else:
            frame = frame

        # Retina Face
        if(verification_step == False):
            frame = cv2.resize(frame, (self.width_size, self.fixed_height))
        
        org_img = frame.copy()
        # Return List of Dictionary of bounding boxes
        faces = self.FD_model.detect(frame, 0.9)
        # Returns Cropped images for all boudinig boxes
        aligned_faces = align_face(faces, frame, align=align)

        # faces is dictionary of bounding boxes
        # aligned_faces is list of cv2 cropped images
        return faces, aligned_faces, org_img  # faces meaning bounding boxes

    def add_visuals(self, image, bounding_boxes, identities, mask_status, ages, emotions):
        '''
        rotate it here
        '''
        for i, box in enumerate(bounding_boxes):
            x = box['facial_area'][0]
            y = box['facial_area'][1]
            w = box['facial_area'][2]
            h = box['facial_area'][3]
            image = cv2.putText(image, identities[i], (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                  
            # image = cv2.putText(image, str(round(ages[i])), (x + 75,y + 15),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            # image = cv2.putText(image, emotions[i], (x + 75, y + 30),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if mask_status[i] == 1:
                image = cv2.putText(image, "Mask", (x, h + 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
                image = cv2.rectangle(image, (w, h), (x, y), (0, 255, 0), 1)

            else: 
                image = cv2.putText(image, "No Mask", (x, h + 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2)
                image = cv2.rectangle(image, (w, h), (x, y), (0, 0, 255), 1)
            
            # image = cv2.rectangle(image, (w, h), (x, y), (0, 255, 0), 1)

        return image

    def add_visuals_verify(self, input, faces):
        '''
        rotate it here
        '''

        for each in list(faces.keys()):
            identity = faces[each]

            facial_area = identity["facial_area"]
            landmarks = identity["landmarks"]

            #highlight facial area
            cv2.rectangle(input, (facial_area[2], facial_area[3])
                , (facial_area[0], facial_area[1]), (255, 0, 0), 2)

            #highlight the landmarks
            cv2.circle(input, tuple(map(int, landmarks["left_eye"])), 1, (0, 0, 255), -1)
            cv2.circle(input, tuple(map(int, landmarks["right_eye"])), 1, (0, 0, 255), -1)
            cv2.circle(input, tuple(map(int, landmarks["nose"])), 1, (0, 0, 255), -1)
            cv2.circle(input, tuple(map(int, landmarks["mouth_left"])), 1, (0, 0, 255), -1)
            cv2.circle(input, tuple(map(int, landmarks["mouth_right"])), 1, (0, 0, 255), -1)

            return input


    def verify(self,img1, img2, metric):

        img1_embedding,_,_ = generate_embedding(img1,self.FR_model,self.model_name["Recognition"],self.age_model,self.emo_model,self.emotion_labels)
        img2_embedding,_,_ = generate_embedding(img2,self.FR_model,self.model_name["Recognition"],self.age_model,self.emo_model,self.emotion_labels)

        if metric == 'cosine':
            distance = dst.findCosineDistance(img1_embedding, img2_embedding)
        elif metric == 'euclidean':
            distance = dst.findEuclideanDistance(img1_embedding, img2_embedding)
        elif metric == 'euclidean_l2':
            distance = dst.findEuclideanDistance(dst.l2_normalize(img1_embedding), dst.l2_normalize(img2_embedding))
        
        #------------------------------
        #verification
        verification = False
        
        threshold = findThreshold(thresholds,metric,self.model_name["Recognition"])

        if distance <= threshold:
            verification = True

        result = {"verification": verification,
                "distance": distance,
                "threshold": threshold,
                }

        return result

    def recognition_process(self, boxes, aligned_faces, metric='cosine'):
        '''
        Use ArcFace or VGG Face

        Including boxes as parameters here in case we decide to use it
        that information for removing side face classification using
        eye distance coordinates.
        '''

        bounding_boxes = []
        identities = []
        mask_status = []
        ages = []
        emotions = []
        for box_id, faces in zip(boxes, aligned_faces):
            t1 = time.time()
            pred_embedding, age, emotion = generate_embedding(faces,
                                                self.FR_model,
                                                self.model_name["Recognition"],
                                                self.age_model,
                                                self.emo_model,
                                                self.emotion_labels
                                                )
            # print(f"embeddings for {box_id} in {time.time()-t1}")
            t1 = time.time()
            target = self.mask_process(faces)
            # print(f"masks for {box_id} in {time.time()-t1}")
            t1 = time.time()
            min_distance = 3465132745
            final_name = "UI"

            for encoding, name in zip(self.known_Encodings, self.known_Names):

                if(target == 1):
                    distance = compare_encodings(pred_embedding, encoding,
                                             metric, self.model_name["Recognition"], # noqa
                                             thresholds_mask)
                    # print(f'Using different threshold for comparing masked faces')
                else:
                    distance = compare_encodings(pred_embedding, encoding,
                                             metric, self.model_name["Recognition"], # noqa
                                             thresholds)
                # print(f"Current distance from {name} : {distance}")
                if(distance == -1):
                    continue
                else:
                    if(distance < min_distance):
                        min_distance = min(distance, min_distance)
                        final_name = name
            bounding_boxes.append(boxes[box_id])
            identities.append(final_name)
            mask_status.append(target)
            ages.append(age)
            emotions.append(emotion)
            # print(f"masks for {box_id} in {time.time()-t1}")
        return bounding_boxes, identities, mask_status, ages, emotions

    # returns mask status 
    def mask_process(self, frame):
        '''
        use network of choice for Mask detection
        '''
        frame = resizingmtcnnfacemask(frame)
        scores = self.Mask_model.predict(frame)[0]
        
        if(scores < 0.5):
            return 1 
        
        return 0

################################################################################################################