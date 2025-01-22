from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import dlib
import cv2
import numpy as np
from io import BytesIO
import requests
from PIL import Image

app = FastAPI()

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

class EmbeddingRequest(BaseModel):
    image_url: str

def process_image(image_url: str):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = np.array(Image.open(BytesIO(response.content)).convert("RGB"))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image download failed: {str(e)}")

@app.post("/embed")
async def get_embedding(request: EmbeddingRequest):
    try:
        img = process_image(request.image_url)
        faces = detector(img)
        if not faces:
            raise HTTPException(status_code=404, detail="No faces detected")
        
        face = faces[0]
        shape = shape_predictor(img, face)
        embedding = face_recognizer.compute_face_descriptor(img, shape)
        
        return {"embedding": list(embedding)}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
