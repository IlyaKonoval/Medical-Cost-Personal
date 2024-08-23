from flask import Flask, request, jsonify
from joblib import load
from explainerdashboard import ExplainerDashboard
import threading
import time
import pandas as pd

app = Flask(__name__)

model = load("/app/final_model_pipeline.joblib")

def run_dashboard():
    db = ExplainerDashboard.from_config("/app/dashboard.yaml")
    db.run(host='0.0.0.0', port=9050, use_waitress=True)

# Запуск Flask-приложения
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction_num = model.predict(df)
    prediction_text = "High" if prediction_num[0] == 1 else "Low"
    response = {'prediction': prediction_text}
    return jsonify(response)

if __name__ == '__main__':
    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.start()

    time.sleep(5)

    app.run(host='0.0.0.0', port=9051, debug=True)