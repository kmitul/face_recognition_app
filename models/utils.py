from numpy import *
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from deepface.commons import functions, distance as dst

def findThreshold(metric):
    if metric == 'cosine':
        return 0.6871912959056619
    elif metric == 'euclidean':
        return 4.1591468986978075
    elif metric == 'euclidean_l2':
        return 1.1315718048269017

def verify(img1, img2, metric, model):

    #representation
    img1_embedding = model.predict(img1)[0]
    img2_embedding = model.predict(img2)[0]

    if metric == 'cosine':
        distance = dst.findCosineDistance(img1_embedding, img2_embedding)
    elif metric == 'euclidean':
        distance = dst.findEuclideanDistance(img1_embedding, img2_embedding)
    elif metric == 'euclidean_l2':
        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_embedding), dst.l2_normalize(img2_embedding))

    #------------------------------
    #verification

    threshold = findThreshold(metric)

    verification = False

    if distance <= threshold:
        verification = True
        print("they are same person")
    else:
        print("they are different persons")

    print("Distance is ",round(distance, 2)," whereas as expected max threshold is ",round(threshold, 2))

    #------------------------------
    #display
    #
    # fig = plt.figure()
    #
    # ax1 = fig.add_subplot(1,2,1)
    # plt.axis('off')
    # plt.imshow(img1[0])
    #
    # ax2 = fig.add_subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(img2[0])
    #
    # plt.show()

    return {'verification': verification,
            'distance': round(disance,2),
            'threshold': round(threshold,2),
            }

def resizingarcface(img_crop_1):
  image_resized = resize(img_crop_1, (112, 112),
                        anti_aliasing=True)

  input_image_1 = image_resized
  input_image_1 = input_image_1.reshape(1,112,112,3)
  return input_image_1

def bounding_box_cropretina(img,bb):
  X = bb['facial_area'][0]
  Y = bb['facial_area'][1]
  W = bb['facial_area'][2]
  H = bb['facial_area'][3]

  cropped_image = img[Y:H, X:W]
  return cropped_image
