# Video Anomaly Detection System

## Overview

This project focuses on developing an **Anomaly Detection System** for surveillance videos using state-of-the-art deep learning models. The goal is to accurately identify and classify abnormal activities such as *Abuse, Arson, Burglary, Fighting, Road Accidents, Shooting*, and more, in real-time.

The project consists of two main parts:
1. **Backend**: Responsible for training, serving the deep learning models, and API communication.
2. **Frontend**: A dashboard for real-time anomaly monitoring and model interaction.

---

## Table of Contents
- [Overview](#overview)
- [Backend](#backend)
  - [Technologies Used](#technologies-used)
  - [Model Details](#model-details)
  - [How to Run the Backend](#how-to-run-the-backend)
- [Frontend](#frontend)
  - [Technologies Used](#technologies-used-1)
  - [Features](#features)
  - [How to Run the Frontend](#how-to-run-the-frontend)
- [Dataset](#dataset)
- [Performance](#performance)
- [Future Improvements](#future-improvements)

---

## Backend

The backend of the system is built using **Flask**, which serves the trained deep learning models and handles requests from the frontend for predictions. The system supports real-time video classification and anomaly detection.

### Technologies Used
- **Python**
- **TensorFlow / Keras**
- **Flask** (Web Framework)
- **OpenCV** (Video Processing)
- **NumPy, Pandas** (Data Processing)

### Model Details

Several models were experimented with, including:

1. **EfficientNetB7 + GRU**  
   - Training Accuracy: 93.29%  
   - Validation Accuracy: 77%  
   - F1 Score: 0.81

2. **MobileNetV2**  
   - Training Accuracy: 93.50%  
   - Validation Accuracy: 58%  
   - F1 Score: 0.69  

3. **InceptionV3**  
   - Training Accuracy: 90.10%  
   - Validation Accuracy: 63%  
   - F1 Score: 0.69  

The **EfficientNetB7 + GRU** architecture yielded the best results in terms of validation accuracy and F1 score, and was chosen as the final model.

### How to Run the Backend

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/video-anomaly-detection.git](https://github.com/KrishnGupta/video-anomaly-detection.git
   cd video-anomaly-detection

2. Install the required dependencies:
    ```
    pip install -r requirements.txt

    ```
3. Run the Flask server:
    ```
    python main.py

The backend will be running on http://127.0.0.1:5000/


## Frontend

The frontend is a **dashboard interface** where users can upload videos or monitor real-time surveillance feeds for anomaly detection. The interface displays confidence scores, performance metrics, and visual plots of accuracy/loss.

### Technologies Used
- **HTML5 / CSS3**
- **JavaScript (vanilla)**
- **Bootstrap 4** (for responsive design)
- **Flask** (for integration with backend)
- **Jinja2** (for templating)

### Features
- **Video Monitoring**: Displays video clips with anomaly classification.
- **Model Performance Visualization**: Shows training accuracy, validation accuracy, and loss plots.
- **User Authentication**: Includes login and registration pages.
- **Real-time Anomaly Alerts**: Users are alerted if an anomaly is detected.

## Dataset

The project uses the **UCF-Crime Dataset**, a large-scale video dataset containing 14 different classes of anomalous activities such as:

- Abuse
- Arson
- Burglary
- Fighting
- Road Accidents
- Shooting
- Stealing
- Arrest
- Assault
- Explosion
- Normal
- Robbery
- Shoplifting
- Vandalism

The dataset is split into normal and anomaly videos for training and testing purposes.

🔗 **Download the dataset**:  
[UCF Crime Dataset](https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?e=3&dl=0)

---

## Performance

Below is a summary of the performance of the three models used in this project:

| Model                   | Training Accuracy | Validation Accuracy | F1 Score |
|-------------------------|-------------------|---------------------|----------|
| **EfficientNetB7 + GRU**| 93.29%            | 77%                 | 0.83     |
| **MobileNetV2**         | 93.50%            | 58%                 | 0.69     |
| **InceptionV3**         | 90.10%            | 63%                 | 0.69     |
--------------------------------------------------------------------------------
## Future Improvements

- **Deployment on Edge Devices**: The system will be optimized for running on edge devices such as surveillance cameras with lower latency.
- **Real-time Streaming**: Integrating a live streaming feed with real-time anomaly detection.
- **Scalability**: Allow the system to monitor multiple cameras simultaneously.
- **Improved UX/UI**: Enhancing the frontend design for a smoother user experience.

---

## License

This project currently does not have a license. You may want to consider adding one in the future to clarify the terms of use and distribution.
