import numpy as np
import pandas as pd
import sklearn
import pickle

import matplotlib.pyplot as plt
import cv2

# Load all models
haar_classifier = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml') # cascade classifier
svm_model =  pickle.load(open('model/model_svm.pickle',mode='rb')) # machine learning model (SVM)
pca_models = pickle.load(open('model/pca_dict.pickle',mode='rb')) # pca dictionary

pca_model = pca_models['pca'] # PCA model
mean_face = pca_models['mean_face'] # Mean Face


def read_image(file_path):
    """Reads an image from a given file path."""
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Error reading image from {file_path}")
    return img

def convert_to_gray(image):
    """Converts an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def crop_face(image, haar_classifier):
    """Crops a face from the given image using a Haar cascade classifier."""
    faces = haar_classifier.detectMultiScale(image, 1.5, 3)
    return faces

def normalize_resize_flatten(image):
    """Normalizes, resizes, and flattens the given face image."""
    roi = image / 255.0
    if image.shape[1] > 100:
        roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)
    else:
        roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)
    return roi_resize.reshape(1, -1)

def subtract_mean(image, mean_face):
    """Subtracts the mean face from the given image."""
    return image - mean_face

def get_eigen_image(pca_model, image):
    """Obtains the eigen image using a PCA model."""
    return pca_model.transform(image)

def inverse_transform_eigen_image(pca_model, eigen_image):
    """Inverse transforms the eigen image."""
    return pca_model.inverse_transform(eigen_image)

def predict_gender(svm_model, eigen_image):
    """Predicts the gender using an SVM model."""
    results = svm_model.predict(eigen_image)
    prob_score = svm_model.predict_proba(eigen_image)
    prob_score_max = prob_score.max()
    return results[0], prob_score_max

def generate_report(result, prob_score_max):
    """Generates a report based on the prediction result."""
    text = "%s : %d" % (result, prob_score_max * 100)
    return text

def draw_prediction_on_image(image, x, y, w, h, result, color, text):
    """Draws the prediction result on the given image."""
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(image, (x, y - 40), (x + w, y), color, -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)

def face_recognition_pipeline(file_path, path = True):
    """Processes an image for gender prediction."""
    if path:
        img = read_image(file_path)
    else:
        img = file_path

    gray = convert_to_gray(img)
    faces = crop_face(gray, haar_classifier)
    
    predictions = []

    for x, y, w, h in faces:
        roi = gray[y:y + h, x:x + w]
        roi_resized = normalize_resize_flatten(roi)
        roi_mean_subtracted = subtract_mean(roi_resized, mean_face)
        eigen_image = get_eigen_image(pca_model, roi_mean_subtracted)
        eig_img = inverse_transform_eigen_image(pca_model, eigen_image)
        result, prob_score_max = predict_gender(svm_model, eigen_image)
        text = generate_report(result, prob_score_max)
        
        if result == 'male':
            color = (255, 255, 0)
        else:
            color = (255, 0, 255)

        draw_prediction_on_image(img, x, y, w, h, result, color, text)

        output = {
            'roi': roi,
            'eig_img': eig_img,
            'prediction_name': result,
            'score': prob_score_max
        }

        predictions.append(output)

    return img, predictions


def display_original_image(img):
    """Displays the original image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def display_prediction_results(predictions):
    """Generates and displays the report with gray scale and eigen images."""
    for i, prediction in enumerate(predictions):
        obj_gray = prediction['roi']  # gray scale
        obj_eig = prediction['eig_img'].reshape(100, 100)  # eigen image
        
        plt.subplot(1, 2, 1)
        plt.imshow(obj_gray, cmap='gray')
        plt.title('Gray Scale Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(obj_eig, cmap='gray')
        plt.title('Eigen Image')
        plt.axis('off')

        plt.show()
        
        print('Predicted Gender =', prediction['prediction_name'])
        print('Predicted score = {:,.2f} %'.format(prediction['score'] * 100))
        
        print('-' * 100)
