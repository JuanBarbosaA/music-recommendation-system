import math
import pandas as pd
import random
import csv
import requests
import base64
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# ================================
# AUTENTICACIÓN SPOTIFY API
# ================================

import os
from dotenv import load_dotenv

load_dotenv()  # carga el archivo .env

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

auth_string = f"{client_id}:{client_secret}"  # Formato requerido por Spotify
b64_auth_string = base64.b64encode(auth_string.encode()).decode()  # Codificación base64

# Solicitud del token de acceso
url_token = "https://accounts.spotify.com/api/token"
headers_token = {"Authorization": f"Basic {b64_auth_string}"}
data_token = {"grant_type": "client_credentials"}
response = requests.post(url_token, headers=headers_token, data=data_token)
access_token = response.json()["access_token"]  # Extrae token del JSON
headers = {"Authorization": f"Bearer {access_token}"}

# ================================
# RECOLECCIÓN DE CANCIONES POR GÉNERO
# ================================
tracks = []  # Lista que almacenará la información de todas las canciones
generos = ["rock", "pop", "clasica", "reggae", "salsa", "vallenato"]
limite_por_genero = 8  # Número de canciones a obtener por género

# Bucle para solicitar canciones de cada género a la API de Spotify
for genero in generos:
    url_search = f"https://api.spotify.com/v1/search?q={genero}&type=track&limit={limite_por_genero}"
    response = requests.get(url_search, headers=headers)
    data = response.json()
    
    # Se extrae información relevante de cada canción
    for item in data["tracks"]["items"]:
        imagen = item["album"]["images"][0]["url"] if item["album"]["images"] else None
        url_spotify = item["external_urls"]["spotify"]
        track_info = {
            "nombre": item["name"],
            "artista": item["artists"][0]["name"],
            "album": item["album"]["name"],
            "genero": genero,
            "id_spotify": item["id"],
            "imagen": imagen,
            "url_spotify": url_spotify
        }
        tracks.append(track_info)

# ================================
# GENERACIÓN DE USUARIOS Y CALIFICACIONES
# ================================
num_usuarios = 3000
usuarios = []
track_genero = [t["genero"] for t in tracks]  # Géneros de todas las canciones

# Se crean usuarios simulados con edad, género, región y género musical favorito
for user_id in range(1, num_usuarios + 1):
    edad = random.randint(15, 60)
    genero_usuario = random.choice(["M", "F"])
    region = random.choice(["Caribe", "Llanos", "Bogotá", "Santander"])
    genero_fav = random.choice(generos)
    clase_usuario = f"Amante del {genero_fav.capitalize()}"

    # Calificaciones de canciones según el gusto del usuario
    calificaciones = []
    for g in track_genero:
        if g == genero_fav:
            calificaciones.append(random.choices([3, 4, 5], weights=[1, 3, 6])[0])
        else:
            calificaciones.append(random.choices([0, 1, 2, 3, 4], weights=[3, 3, 2, 1, 1])[0])

    usuario = [user_id, edad, genero_usuario, region, genero_fav.capitalize(), clase_usuario] + calificaciones
    usuarios.append(usuario)

# ================================
# AÑADIR RUIDO A LOS USUARIOS
# ================================
# Mezcla los usuarios y cambia el género favorito de un 30% de ellos
random.shuffle(usuarios)
porcentaje_ruido = 0.3
num_ruido = int(num_usuarios * porcentaje_ruido)
for i in range(num_ruido):
    actual_genero = usuarios[i][4]
    posibles_generos = [g.capitalize() for g in generos if g.capitalize() != actual_genero]
    nuevo_genero = random.choice(posibles_generos)
    usuarios[i][4] = nuevo_genero
    usuarios[i][5] = f"Amante del {nuevo_genero}"

# ================================
# GUARDAR DATASETS EN CSV
# ================================
# Archivo de usuarios y calificaciones
header = ["UserID", "Edad", "Genero", "Region", "GeneroFav", "ClaseUsuario"] + [t["nombre"] for t in tracks]
with open("dataset_canciones_spotify.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(usuarios)

# Archivo de información de canciones
with open("info_canciones_spotify.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Nombre", "Artista", "Álbum", "Género", "SpotifyID", "Imagen", "URL_Spotify"])
    for t in tracks:
        writer.writerow([t["nombre"], t["artista"], t["album"], t["genero"], t["id_spotify"], t["imagen"], t["url_spotify"]])

# ================================
# FUNCIONES DE SIMILITUD Y KNN
# ================================
def _common_indices(vec_a, vec_b):
    """Devuelve índices donde ambos vectores tienen valores > 0"""
    return [i for i, (a, b) in enumerate(zip(vec_a, vec_b)) if a > 0 and b > 0]

def cosine_similarity(a, b):
    """Calcula similitud coseno entre dos vectores"""
    idx = _common_indices(a, b)
    if not idx:
        return 0.0
    dot = sum(a[i] * b[i] for i in idx)
    norm_a = math.sqrt(sum(a[i] ** 2 for i in idx))
    norm_b = math.sqrt(sum(b[i] ** 2 for i in idx))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

METRIC_FUNCTIONS = {"cosine": cosine_similarity}

# ================================
# FUNCIONES PARA CARGAR DATASET
# ================================
def load_dataset(csv_path):
    """Carga dataset de usuarios y devuelve DataFrame y lista de canciones"""
    df = pd.read_csv(csv_path)
    song_cols = list(df.columns[6:])
    return df, song_cols

def load_song_info(csv_path):
    """Carga dataset de información de canciones"""
    return pd.read_csv(csv_path)

def user_vector_from_row(row, song_cols):
    """Convierte fila de usuario a vector de calificaciones"""
    return [float(row[c]) for c in song_cols]

def build_user_vectors(df_users, song_cols):
    """Crea listas y mapas de vectores de usuarios"""
    user_list, user_map = [], {}
    for _, row in df_users.iterrows():
        uid = int(row['UserID'])
        vec = user_vector_from_row(row, song_cols)
        user_list.append((uid, vec))
        user_map[uid] = vec
    return user_list, user_map

# ================================
# SIMILITUDES Y VECINOS
# ================================
def compute_all_similarities(candidate_vec, user_vectors, metric="cosine"):
    """Calcula similitud de un candidato con todos los usuarios"""
    
    func = METRIC_FUNCTIONS[metric]  # Selecciona la función de similitud (cosine)
    results = []  # Lista para guardar (user_id, similitud)
    
    
    # Recorre todos los usuarios del dataset
    for user_id, vec in user_vectors:
        sim = func(candidate_vec, vec)  # Calcula la similitud entre candidato y usuario
        results.append((user_id, sim))  # Guarda el id del usuario y la similitud
    
    results.sort(key=lambda x: x[1], reverse=True)  # Ordena de mayor a menor similitud
    return results  # Devuelve la lista de todos los usuarios con su similitud


def get_top_k_neighbors(similarities, k):
    """Obtiene los k vecinos más similares con similitud > 0"""
    
    filtered = [p for p in similarities if p[1] > 0]  # Filtra solo usuarios con similitud positiva
    return filtered[:k]  # Devuelve los k primeros (los más similares)


def classify_candidate_by_neighbors(neighbors, df_users):
    """Clasifica al usuario candidato según los vecinos más cercanos"""
    votes = []  # Lista para guardar los géneros favoritos de los vecinos
    # Recorre los vecinos más cercanos (user_id y su similitud)
    for user_id, _ in neighbors:
        row = df_users.loc[df_users['UserID'] == user_id]  # Obtiene la fila del vecino en el DataFrame
        if not row.empty:  # Si se encuentra el usuario
            votes.append(str(row.iloc[0]['GeneroFav']))  # Añade su género favorito a la lista de votos
    # Si ningún vecino válido encontró género, retorna mensaje
    if not votes:
        return "Sin vecinos válidos", {}
    counter = Counter(votes)  # Cuenta cuántos votos tiene cada género
    # Devuelve el género más votado y un diccionario con el conteo de todos los géneros
    return counter.most_common(1)[0][0], dict(counter)


def predict_ratings(candidate_vec, neighbors, user_map, song_cols):
    """Predice calificaciones de canciones que el usuario candidato aún no ha calificado"""
    predictions = {}  # Diccionario donde guardaremos las canciones y su calificación predicha
    # Recorre todas las canciones del dataset
    for idx, song in enumerate(song_cols):
        if candidate_vec[idx] > 0:
            # Si el usuario ya calificó la canción, se salta y no predice
            continue
        # Inicializa acumuladores para el promedio ponderado
        num, den, count_valid = 0.0, 0.0, 0
        # Recorre los vecinos más cercanos y su similitud
        for user_id, sim in neighbors:
            rating = user_map[user_id][idx]  # Obtiene la calificación del vecino para esta canción
            if rating > 0:  # Solo considera vecinos que hayan calificado la canción
                num += sim * rating  # Multiplica la calificación por la similitud y suma al numerador
                den += abs(sim)       # Suma la similitud absoluta al denominador (ponderación)
                count_valid += 1      # Cuenta cuántos vecinos válidos contribuyeron
        # Si al menos un vecino contribuyó y el denominador no es cero
        if count_valid > 0 and den > 0:
            predictions[song] = num / den  # Calcula promedio ponderado y lo guarda
    return predictions  # Devuelve diccionario con canciones y calificaciones predichas


def recommend_songs(predictions, song_info_df, top_n=10, user_genre=None):
    """Genera recomendaciones top N con información de canciones"""
    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    enriched = []
    song_info_df['Nombre'] = song_info_df['Nombre'].str.strip().str.lower()
    for song_name, score in ranked:
        song_name_clean = song_name.strip().lower()
        info_row = song_info_df[song_info_df['Nombre'] == song_name_clean]
        if info_row.empty:
            continue
        info = info_row.iloc[0]
        enriched.append({
            "Nombre": info['Nombre'],
            "Artista": info['Artista'],
            "Album": info['Álbum'],
            "Genero": info['Género'],
            "Score": score,
            "Imagen": info['Imagen'],
            "URL": info['URL_Spotify']
        })
        if len(enriched) >= top_n:
            break
    return enriched

def actualizar_calificacion(usuario, cancion, puntaje):
    """Actualiza la calificación de una canción para un usuario"""
    if cancion in usuario:
        usuario[cancion] = puntaje

def recalcular_perfil(usuario_candidato, df_users, user_map, song_info_df, song_cols, k=5, top_n=10):
    """Recalcula género predicho y recomendaciones después de nuevas calificaciones"""
    candidate_vec = [usuario_candidato[s] for s in song_cols]
    user_list_filtered = [(uid, vec) for uid, vec in build_user_vectors(df_users, song_cols)[0]]
    sims = compute_all_similarities(candidate_vec, user_list_filtered)
    neighbors = get_top_k_neighbors(sims, k)
    predicted_class, votes = classify_candidate_by_neighbors(neighbors, df_users)
    predictions = predict_ratings(candidate_vec, neighbors, user_map, song_cols)
    recommendations = recommend_songs(predictions, song_info_df, top_n=top_n, user_genre=predicted_class)
    return predicted_class, recommendations

# ================================
# EVALUACIÓN KNN
# ================================
def evaluate_knn_with_split(csv_path, test_ratio=0.2, k=5, metric="cosine"):
    """Evalúa el modelo KNN usando un conjunto de prueba"""
    # Carga el dataset de usuarios y canciones
    df, song_cols = load_dataset(csv_path)
    # Mezcla aleatoriamente los usuarios para dividir en entrenamiento y prueba
    users = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Calcula el índice de separación según el ratio de prueba (por ejemplo 20%)
    split_index = int(len(users) * (1 - test_ratio))
    # Datos de entrenamiento y prueba
    df_train = users.iloc[:split_index]  # primeros 80% → entrenamiento
    df_test = users.iloc[split_index:]   # últimos 20% → prueb
    # Construye los vectores de calificaciones de los usuarios de entrenamiento
    train_vectors, train_map = build_user_vectors(df_train, song_cols)
    # Listas para almacenar los géneros verdaderos y predichos
    y_true, y_pred = [], []
    # Itera sobre cada usuario del conjunto de prueba
    for _, row in df_test.iterrows():
        candidate_vec = user_vector_from_row(row, song_cols)  # Vector de calificaciones del candidato
        sims = compute_all_similarities(candidate_vec, train_vectors, metric=metric)  # Similitudes con todos los usuarios de entrenamiento
        neighbors = get_top_k_neighbors(sims, k)  # Selecciona los K vecinos más similares
        predicted_class, _ = classify_candidate_by_neighbors(neighbors, df_train)  # Predice el género del candidato
        # Solo se guarda si hay vecinos válidos
        if predicted_class != "Sin vecinos válidos":
            y_true.append(row["GeneroFav"])  # Género real del usuario de prueba
            y_pred.append(predicted_class)   # Género predicho por KNN


    # Métricas de evaluación
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(df["GeneroFav"].unique()))
    print(f"Accuracy global: {accuracy:.4f}")
    print(f"F1-score global: {f1:.4f}")
    print("Matriz de confusión:")
    print(cm)
    print("\n--- Precision y Recall por clase ---")
    print(classification_report(y_true, y_pred, zero_division=1))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(df["GeneroFav"].unique()))
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusión - Conjunto de Prueba (20%)")
    plt.show()

# ================================
# EJEMPLO DE USO DEL SISTEMA
# ================================
# Cargar datasets
df_users, song_cols = load_dataset("dataset_canciones_spotify.csv")
song_info_df = load_song_info("info_canciones_spotify.csv")
user_list, user_map = build_user_vectors(df_users, song_cols)

# Crear un usuario candidato con calificaciones iniciales
usuario_candidato = {song: 0 for song in song_cols}

# Simular calificaciones nuevas
calificaciones_nuevas = [
    ("Rock That Body", 5),
    ("Pop Muzik", 3),
    ("Roots, Rock, Reggae", 4),
    ("Salsa Con Coco", 5)
]

for cancion, puntaje in calificaciones_nuevas:
    actualizar_calificacion(usuario_candidato, cancion, puntaje)
    genero_predicho, recomendaciones = recalcular_perfil(
        usuario_candidato, df_users, user_map, song_info_df, song_cols
    )
    print(f"\nDespués de calificar '{cancion}' con {puntaje}:")
    print(f"Género detectado: {genero_predicho}")
    print("Top 5 recomendaciones:")
    for rec in recomendaciones[:10]:
        print(f"- {rec['Nombre']} - {rec['Artista']} ({rec['Genero']}) Score: {rec['Score']:.2f}")
        print(f"  URL: {rec['URL']}")
        print(f"  Imagen: {rec['Imagen']}")

# Evaluación final del modelo KNN
print("\n🔹 Evaluación KNN con conjunto de prueba (20%)")
evaluate_knn_with_split("dataset_canciones_spotify.csv", test_ratio=0.2, k=5, metric="cosine")
