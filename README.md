Gesture Detection using Machine Learning

Overview
This project implements a gesture detection system using a machine learning model trained on hand gesture images. It utilizes OpenCV for image processing, MediaPipe for hand landmark detection, and Scikit-Learn for training a Random Forest classifier.

Features
- Captures hand gesture images using OpenCV.
- Extracts hand landmarks using MediaPipe.
- Trains a Random Forest classifier for gesture recognition.
- Performs real-time gesture classification.

Technologies Used
- Python
- OpenCV
- MediaPipe
- Scikit-Learn
- NumPy

Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gesture-detection.git
   cd gesture-detection
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Dataset Collection
Run the following command to start data collection:
```bash
python collect_data.py
```
The script will capture images for different gesture classes and store them in the `data/` directory.

Training the Model
To train the gesture recognition model, run:
```bash
python train_model.py
```
This will train a Random Forest classifier and save the model as `model.p`.

Running the Gesture Detection System
Execute the following command to start real-time gesture recognition:
```bash
python recognize_gestures.py
```
The program will open a webcam feed and classify hand gestures in real time.

Model Performance
The accuracy of the trained model is displayed after training. It can be improved by collecting more data and fine-tuning the model parameters.

Future Improvements
- Implement deep learning models for improved accuracy.
- Add more hand gestures.
- Improve real-time processing speed.


