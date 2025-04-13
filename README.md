
# VibeDetect - Voice Emotion Detection System

**Author:** Bavitha Mamidi

---

## Description

VibeDetect is a machine learning-based voice emotion recognition system that detects emotional states from audio speech samples using audio signal processing techniques. It utilizes the RAVDESS dataset to train a model capable of classifying emotions such as **calm**, **happy**, **fearful**, and **disgust** from audio recordings.

This system uses a combination of MFCC, Chroma, and Mel spectrogram features extracted from audio signals and feeds them into a Multi-Layer Perceptron (MLP) classifier for training and prediction. VibeDetect helps lay the foundation for emotionally-aware systems in fields like healthcare, virtual assistants, and customer service.

---

## Features

- Processes audio files from the RAVDESS dataset  
- Extracts relevant audio features (MFCCs, Chroma, Mel Spectrogram)  
- Trains a neural network using `MLPClassifier` from `scikit-learn`  
- Predicts emotional states from new/unseen audio samples  
- Supports reusable model saving and loading with `pickle`  
- Includes a dedicated class `SpeechEmotionAnalyzer` for modular prediction  

---

## Technologies Used

- Python  
- Librosa  
- NumPy  
- Scikit-learn  
- Pandas  
- Pickle  
- SoundFile  

---

## Dataset

**Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**

- 24 actors with multiple emotional expressions  
- File format: WAV  

### Supported Emotions (observed):
- calm  
- happy  
- fearful  
- disgust  

---

## Requirements

Install the following dependencies before running the project:

```bash
pip install numpy librosa pandas scikit-learn soundfile audioread
```

---

## Usage Instructions

### 1. Set the dataset path  
Make sure the RAVDESS dataset is placed inside the project folder under `Audio_Speech_Actors_01-24`

### 2. Extract Features and Train the Model  
The script loads audio files, extracts features, and splits the dataset into training and testing sets. It then trains a neural network using the extracted features.

### 3. Model Saving and Loading  
The trained model is saved using `pickle` in a `.sav` file (`modelForPrediction1.sav`). This model can later be loaded for real-time predictions without retraining.

### 4. Prediction  
Use the `SpeechEmotionAnalyzer` class to load the saved model and predict the emotion of any `.wav` audio file:

```python
analyzer = SpeechEmotionAnalyzer('modelForPrediction1.sav')
prediction = analyzer.predict_emotion('path_to_audio.wav')
print(prediction)
```

---

## Model Performance

- **Accuracy:** ~61.98% on test data  

### F1 Scores:
- **Calm:** 0.68  
- **Happy:** 0.62  
- **Fearful:** 0.58  
- **Disgust:** 0.61  

*Performance can be further improved with more balanced training data, data augmentation, or using deep learning architectures.*

---

## File Structure Overview

```
Audio_Speech_Actors_01-24/        # RAVDESS dataset folder  
modelForPrediction1.sav           # Trained MLP model file  
SpeechEmotionAnalyzer             # Python class for loading the model and predicting emotions  
(training and testing scripts)    # Data loading, feature extraction, model training, evaluation
```

---

## Future Enhancements

- Expand to support more emotion classes from the RAVDESS dataset  
- Integrate with a real-time microphone input for live emotion detection  
- Develop a user-friendly UI or web interface  
- Try deep learning models (e.g., CNN, RNN, LSTM) for improved accuracy  
- Explore multilingual emotion detection  
