import logging
from flask import Flask, request, jsonify
from model import My_Rec_Model  

# Настройка логирования
logging.basicConfig(
    filename="app.log",  # Имя файла для логов
    level=logging.INFO,   # Уровень логирования
    format="%(asctime)s - %(levelname)s - %(message)s",  # Формат сообщения
)

app = Flask(__name__)

# Создаём объект модели при запуске
model = My_Rec_Model()
model.warmup()
logging.info("Application started, Model instance created.")

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    movies = data.get("movies")
    ratings = data.get("ratings")
    n = data.get("n")
    if not n:
        n = 20
    rec_ids, rec_ratings = model.predict(movies, ratings, n)
    rec_names = model.get_movie_titles(rec_ids)
    return jsonify(rec_names=rec_names, rec_ratings=rec_ratings)

@app.route('/api/reload', methods=['POST'])
def reload():
    model.warmup()
    logging.info("Application started, Model warmed up.")

@app.route('/api/similar', methods=['POST'])
def similar():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    name = data.get("movie_name")
    answer = model.similar_for_name(name)
    return jsonify(sim_names=answer)