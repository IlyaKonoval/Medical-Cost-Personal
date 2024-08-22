FROM python:3.10-slim-buster

# Устанавливаем рабочую директорию
WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip setuptools \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/
COPY preprocessed_insurance.csv /app/
COPY final_model_pipeline.joblib /app/
COPY dashboard.yaml /app/
COPY explainer.dill /app/


EXPOSE 9050
EXPOSE 9051


CMD ["python", "/app/app/app.py"]