


# Face Recognition App

A versatile web application build using state of the art Face detection and recognition AI, providing you with following features :  

- [:heavy_check_mark:] Face verification 
- [:heavy_check_mark:] Face matching in the database
- [:heavy_check_mark:] Attendence Marking 

## Overview

<img src = "https://github.com/kmitul/face_recognition_app/blob/main/Extras/pipeline.png">

This was the project mainly aimed to construct a robust face recognition application with wide applications using state of the art facial detection and recognition techniques. 
We have implemented various modules that construct the whole system, these modules are the following :

1. Face analysis module 
2. Web-Application module
3. Database Module  

### Face Analysis Module
We designed a complete pipeline with detector backend as **Retina-Face** for detecting multiple faces present in the frame. We used **Arc-Face** and **VGG-Face** as models for generating face vectors. For mask detection we used **Xception** architecture and emotion and age detection model uses **VGG-Face** and **Alex-net** based architectures respectively.  

### Web-Application Module

We built a **Flask** Server Backend which runs the whole application by    integrating all the AI models, Database, Sheet APIs and frontend HTML files and scripts together.​ We used **MDBootstrap** framework whic handles all the UI designs of the      application.​ 
**Javascripts** for establishing the duplex communication between client and the 	   	   	
server by accessing client-side webcam and sending snaps as post request to the backend.

### Database Module

We stored facial embeddings of employees as JSON data.​ We Stored the attendance records of employees in Google Sheets.​ Used Sheets API for integration with the server.​ Finally extended the Google Sheet integration with **Tableau** dashboard.

# Setup 

Run following commands in terminal opened in the project directory to run the app.

Create a new virtual environment and install the required packages : 
```
pip install -r requirements.txt
```

## Download Pretrained Model Weights

Download the weights from the google drive link below and save them into pipeline/weights directory before running the code.

### Face Detectors 

**1. Retina-Face** 

Run the following command:
```
gdown --id 1oyNYwGvnCT1HOIOQq6yZ_uX06GBXzMCw
```
Alternatively, the link to the same is given below:
```
https://drive.google.com/file/d/1oyNYwGvnCT1HOIOQq6yZ_uX06GBXzMCw/view?usp=sharing
```

### Face Recognition

**1. VGG-Face** 

Run the following command:
```
gdown --id 1nuLihFS61FCGotF2KRcCzOqLhrr6wPAW
```
Alternatively, the link to the same is given below:
```
https://drive.google.com/file/d/1nuLihFS61FCGotF2KRcCzOqLhrr6wPAW/view?usp=sharing
```
**2. ArcFace** 

Run the following command:
```
gdown --id 1atHsxw9XE1oxeipr008EkImy5n6-K-NR
```
Alternatively, the link to the same is given below:
```
https://drive.google.com/file/d/1atHsxw9XE1oxeipr008EkImy5n6-K-NR/view?usp=sharing
```

### Age Model

Run the following command:
```
gdown --id 1X5c_SGcOEhfrSjvGaIYqtQk0J0sG80xu
```
Alternatively, the link to the same is given below:
```
https://drive.google.com/file/d/1X5c_SGcOEhfrSjvGaIYqtQk0J0sG80xu/view?usp=sharing
```

### Emotion Recogntion Model

Run the following command:
```
gdown --id 1YPrAuQ1_CpVhloXXXa8QuTrFk5KE76Id
```
Alternatively, the link to the same is given below:
```
https://drive.google.com/file/d/1YPrAuQ1_CpVhloXXXa8QuTrFk5KE76Id/view?usp=sharing
```

## How to use

Run the app : 
```
python app.py
```
