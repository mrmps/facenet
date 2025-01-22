FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 \
    && bunzip2 shape_predictor_68_face_landmarks.dat.bz2 \
    && mv shape_predictor_68_face_landmarks.dat /root/

RUN wget https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2 \
    && bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2 \
    && mv dlib_face_recognition_resnet_model_v1.dat /root/

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas0 \
    libgomp1 \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y build-essential cmake \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/shape_predictor_68_face_landmarks.dat .
COPY --from=builder /root/dlib_face_recognition_resnet_model_v1.dat .
COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]