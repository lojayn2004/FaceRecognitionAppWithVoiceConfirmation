import joblib
import numpy as np
import os

class FaceRecognizer:
    def __init__(self, model_path="models", unknown_threshold=0.28):
       
        self.unknown_threshold = unknown_threshold
        
        self.svm = joblib.load(os.path.join(model_path, 'svm_model.pkl'))
        self.le = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
      
    def recognize(self, embedding):
        probabilities = self.svm.predict_proba(embedding)[0]
        best_idx = np.argmax(probabilities)
        confidence = probabilities[best_idx]
      
    
        if confidence < self.unknown_threshold:
            return "UNKNOWN", confidence
        else:
            name = self.le.inverse_transform([best_idx])[0]
        return name, confidence
    
    def set_threshold(self, threshold):
        if 0.1 <= threshold <= 0.9:
            self.unknown_threshold = threshold
           