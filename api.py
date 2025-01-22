import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Create the Modal app first
stub = modal.App("dlib-face-embedding")

# Then create the FastAPI app
app = FastAPI()

# Define container image
image = modal.Image.debian_slim(python_version="3.10").apt_install(
    "wget",
    "build-essential", 
    "cmake"
).pip_install(
    "deepface",
    "opencv-python-headless",
    "pillow", 
    "requests"
)

class EmbeddingRequest(BaseModel):
    image_url: str

@stub.function(image=image, gpu=False)
@modal.asgi_app()
def fastapi_app():
    from deepface import DeepFace
    import cv2
    import numpy as np
    from PIL import Image
    from io import BytesIO
    import requests
    
    model = DeepFace.build_model("Dlib")

    @app.post("/embed")
    async def get_embedding(request: EmbeddingRequest):
        try:
            response = requests.get(request.image_url)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_np = np.array(img)
            img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            embedding_objs = DeepFace.represent(
                img_path=img_cv2,
                model_name="Dlib",
                detector_backend="opencv",
                enforce_detection=False
            )

            if not embedding_objs:
                raise HTTPException(status_code=404, detail="No face detected")
                
            return {"embedding": embedding_objs[0]["embedding"]}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app