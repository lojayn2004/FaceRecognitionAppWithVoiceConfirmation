import torch
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import numpy as np

class FaceDetector:
    def __init__(self, device='auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Face Detector using: {self.device}")
        

        self.mtcnn = MTCNN(device=self.device)
    
    def detect_faces(self, frame):
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        
        boxes, probs = self.mtcnn.detect(pil_image)
        
        valid_faces = []
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob > 0.95:  
                    x1, y1, x2, y2 = map(int, box)
                    
                    if (x2 - x1) >= 50 and (y2 - y1) >= 50:
                        face_region = rgb_frame[y1:y2, x1:x2]
                        if face_region.size > 0:
                            valid_faces.append({
                                'bbox': (x1, y1, x2, y2),
                                'face_region': face_region,
                                'confidence': prob
                            })
        
        return valid_faces
    

    def extract_face_embedding(self, face_region, resnet_model):
        try:
            resnet_model.eval()
            resnet_model.to(self.device)

            face_pil = Image.fromarray(face_region)
            face_tensor = self.mtcnn(face_pil)
            if face_tensor is not None:
                if face_tensor.dim() == 3:
                   face_tensor = face_tensor.unsqueeze(0) 
                elif face_tensor.dim() == 5:
                   face_tensor = face_tensor.squeeze(1) 
            
            
                face_tensor = face_tensor.to(self.device).float()
                with torch.no_grad():
                   embedding = resnet_model(face_tensor.to(self.device)).cpu().numpy()
                
                 
                return embedding
        except Exception as e:
                print(f"Error extracting embedding: {e}")
        return None