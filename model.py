import numpy as np
import argparse
import csv
from sklearn.metrics.pairwise import cosine_similarity

def build_movie_matrix(r_train):
    #Создает матрицу пользователь-фильм и маппинг ID фильмов.
    
    unique_users = np.unique(r_train[:, 0])
    unique_movies = np.unique(r_train[:, 1])

    user_to_index = {user_id: i for i, user_id in enumerate(unique_users)}
    movie_to_index = {movie_id: i for i, movie_id in enumerate(unique_movies)}
    index_to_movie = {i: movie_id for movie_id, i in movie_to_index.items()}

    num_users = len(unique_users)
    num_movies = len(unique_movies)

    matrix = np.zeros((num_users, num_movies))

    for user_id, movie_id, rating in r_train:
        matrix[user_to_index[int(user_id)], movie_to_index[int(movie_id)]] = rating
        
    matrix_norm = normalize_user_ratings_zscore(matrix)

    return matrix_norm, movie_to_index, index_to_movie

def compute_movie_similarity(movie_matrix):
    #Вычисляет матрицу сходства между фильмами
    movie_matrix = movie_matrix.T
    similarity = cosine_similarity(movie_matrix)  
    return similarity

def normalize_user_ratings_zscore(movie_matrix):
    #Нормализует оценки пользоветелей.
    #movie_matrix: Матрица пользователь-фильм с оценками
    user_means = np.mean(movie_matrix, axis=1) 
    user_stddevs = np.std(movie_matrix, axis=1)
    movie_matrix_normalized = (movie_matrix - user_means[:, np.newaxis]) / user_stddevs[:, np.newaxis]  # Вычитание среднего и деление на стандартное отклонение
    return movie_matrix_normalized


class My_Rec_Model:
    def warmup(self):
        with open("data/model/movie_matrix.csv", "r") as f:
            self.movie_matrix = np.loadtxt(f, delimiter=";")
        with open("data/model/movie_to_index.csv", "r") as f:
            reader = csv.reader(f)
            self.movie_to_index = {int(row[0]): int(row[1]) for row in reader}
        with open("data/model/index_to_movie.csv", "r") as f:
            reader = csv.reader(f)
            self.index_to_movie = {int(row[0]): int(row[1]) for row in reader}
        with open("data/model/sim_matrix.csv", "r") as f:
            self.similarity_matrix = np.loadtxt(f, delimiter=";")
        with open("data/model/movies.dat", "r") as file:
            content = file.read()
            content = content.replace("::", ";")
        with open("data/model/movies.dat", "w") as file:
            file.write(content)
        self.movies = np.genfromtxt("data/model/movies.dat", dtype=str, delimiter=";", invalid_raise=False)

    def train(self, file_path):
        train_file = open(file_path, "r")
        r_train = np.loadtxt(train_file, dtype='int', delimiter=";")
        n_users = len(np.unique(r_train[:, 0]))
        n_movies = len(np.unique(r_train[:, 1]))
        r_train = r_train[:, :3]
        
        movie_matrix, movie_to_index, index_to_movie = build_movie_matrix(r_train)
        np.savetxt("data/model/movie_matrix.csv", movie_matrix, delimiter=";")
        with open("data/model/movie_to_index.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for key, value in movie_to_index.items():
                writer.writerow([key, value])
        with open("data/model/index_to_movie.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for key, value in index_to_movie.items():
                writer.writerow([key, value])
        similarity_matrix = compute_movie_similarity(movie_matrix)
        np.savetxt("data/model/sim_matrix.csv", similarity_matrix, delimiter=";")
        self.warmup()

    def recommend(self, movie_ids, ratings, top_n):
        #Рекомендует фильмы на основе списка фильмов и оценок пользователя.
        #movie_ids: список id фильмов, которые оценил пользователь
        #ratings: список оценок пользователя
        #movie_matrix: матрица пользователь-фильм
        #similarity_matrix: матрица сходства между фильмами
        #movie_to_index: словарь {movie_id: номер}
        #index_to_movie: словарь {номер: movie_id}
        #top_n: количество рекомендаций
        
        scores = np.zeros(self.movie_matrix.shape[1])

        for movie, rating in zip(movie_ids, ratings):
            if movie in self.movie_to_index:  # Пропускаем фильмы, которых нет в базе
                index = self.movie_to_index[movie]
                scores += self.similarity_matrix[index] * rating

        for movie in movie_ids:
            if movie in self.movie_to_index:
                scores[self.movie_to_index[movie]] = -np.inf  # Исключаем уже оцененные фильмы

        recommended_indices = np.argsort(scores)[-top_n:][::-1]
        return [self.index_to_movie[idx] for idx in recommended_indices]
    
    def predict_rating(self, movie_id, user_movies, user_ratings):
        if movie_id not in self.movie_to_index:
            return np.mean(user_ratings)  # Если фильма нет в данных, возвращаем средний рейтинг пользователя

        movie_idx = self.movie_to_index[movie_id]
        numer = 0  # Числитель (взвешенная сумма оценок)
        denom = 0  # Знаменатель (сумма модулей схожести)

        for user_movie, rating in zip(user_movies, user_ratings):
            if user_movie in self.movie_to_index:
                idx = self.movie_to_index[user_movie]
                sim = self.similarity_matrix[movie_idx, idx]
                numer += sim * rating
                denom += abs(sim)

        return numer / denom if denom != 0 else np.mean(user_ratings)

    def predict(self, movie_ids, ratings, n):
        recommendations = self.recommend(movie_ids, ratings, top_n = n)
        rec_ratings = []
        
        for movie_id in recommendations:
            rec_rating = self.predict_rating(movie_id, movie_ids, ratings)
            rec_ratings.append(rec_rating)
        
        return recommendations, rec_ratings
    
    def get_movie_titles(self, ids):
        # Возвращает два списка: найденные movie_ids и их названия movie_titles
        #:param ids: list, список ID фильмов
        #:return: (list, list) - два списка (ID, Названия)

        movie_dict = {int(row[0]): row[1] for row in self.movies}  # {id: название}
        
        found_ids = []
        found_titles = []

        for movie_id in ids:
            title = movie_dict.get(movie_id)
            if title:  # Если название найдено
                found_ids.append(movie_id)
                found_titles.append(title)

        return found_titles
    
    def get_similar_movies(self, movie_id, top_n):
        # Возвращает top_n фильмов, наиболее похожих на заданный movie_id
        #:param movie_id: int, ID фильма
        #:param top_n: int, количество рекомендаций
        #:return: list, список ID похожих фильмов

        if movie_id not in self.movie_to_index:
            return []  # Если фильма нет в базе, возвращаем пустой список

        index = self.movie_to_index[movie_id]  # Находим индекс фильма
        similarity_scores = self.similarity_matrix[index]  # Берём строку из матрицы сходства
        similar_indices = np.argsort(similarity_scores)[-top_n-1:-1][::-1]  # Берём топ N (без самого фильма)
        
        return self.get_movie_titles([self.index_to_movie[idx] for idx in similar_indices])
    
    def similar_for_name(self, title, n=5):
        # Возвращает похожие для фильма по названию
        #:param title: str, название фильма

        for movie_id, movie_title in self.movies:
            if movie_title == title:
                return self.find_similar(movie_id, n)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument('mode', choices=['train', 'evaluate'], help="Выбор режима: train или evaluate")
    parser.add_argument('--dataset', type=str, required=True, help="Путь к набору данных")

    args = parser.parse_args()

    model = My_Rec_Model()

    if args.mode == 'train':
        model.train(args.dataset)