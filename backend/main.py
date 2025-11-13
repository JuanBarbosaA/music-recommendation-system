# backend/main.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
import pandas as pd
import math
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ======================
# 🔓 CORS
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# 📦 Cargar dataset y normalizar nombres
# ======================
df_users = pd.read_csv("dataset_canciones_spotify.csv")
song_info_df = pd.read_csv("info_canciones_spotify.csv")

# Normalizar nombres de canciones y columnas
song_cols = [c.strip() for c in df_users.columns[6:]]
df_users.columns = [c.strip() for c in df_users.columns]
song_info_df["Nombre"] = song_info_df["Nombre"].str.strip()

# Crear mapa de usuarios
user_map = {int(row['UserID']): [float(row[c]) for c in song_cols] for _, row in df_users.iterrows()}

# ======================
# 🧠 Funciones auxiliares
# ======================
def build_user_vectors(df, song_cols):
    user_list = []
    for _, row in df.iterrows():
        uid = int(row['UserID'])
        vec = [float(row[c]) for c in song_cols]
        user_list.append((uid, vec))
    return user_list

user_list = build_user_vectors(df_users, song_cols)

def cosine_similarity(a, b):
    # Considera todos los índices, reemplazando ceros por pequeños valores para evitar ceros totales
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(y ** 2 for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

def get_top_k_neighbors(candidate_vec, k=5):
    sims = [(uid, cosine_similarity(candidate_vec, vec)) for uid, vec in user_list]
    sims = [s for s in sims if s[1] > 0]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

def classify_candidate(candidate_vec, k=5):
    neighbors = get_top_k_neighbors(candidate_vec, k)
    votes = []
    for uid, _ in neighbors:
        row = df_users.loc[df_users["UserID"] == uid]
        if not row.empty:
            votes.append(row.iloc[0]["GeneroFav"])
    if not votes:
        return "Sin determinar"
    return Counter(votes).most_common(1)[0][0]

def recommend_songs(candidate_vec, k=5, top_n=10, user_genre=None):
    neighbors = get_top_k_neighbors(candidate_vec, k)
    predictions = {}
    for idx, song in enumerate(song_cols):
        if candidate_vec[idx] > 0:
            continue
        num, den = 0, 0
        for uid, sim in neighbors:
            rating = user_map[uid][idx]
            if rating > 0:
                num += sim * rating
                den += abs(sim)
        if den > 0:
            predictions[song] = num / den

    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    enriched = []
    for song_name, score in ranked:
        row = song_info_df[song_info_df["Nombre"].str.lower() == song_name.lower()]
        if row.empty:
            continue
        r = row.iloc[0]
        enriched.append({
            "Nombre": r["Nombre"],
            "Artista": r["Artista"],
            "Album": r.get("Álbum", ""),
            "Genero": r["Género"],
            "Score": score,
            "URL": r["URL_Spotify"],
            "Imagen": r.get("Imagen", None)
        })
        if len(enriched) >= top_n:
            break
    return enriched

# ======================
# 🎧 Modelo Pydantic
# ======================
class Candidate(BaseModel):
    ratings: dict  # {"Rock That Body": 5, "Pop Muzik": 3, ...}

# ======================
# 🚀 Endpoints
# ======================

@app.get("/api/songs")
def get_all_songs():
    songs = []
    for _, row in song_info_df.iterrows():
        songs.append({
            "Nombre": row["Nombre"],
            "Artista": row["Artista"],
            "Album": row["Álbum"] if "Álbum" in row else row.get("Album", ""),
            "Genero": row["Género"],
            "URL": row["URL_Spotify"],
            "Portada": row["Imagen"] if "Imagen" in row else None
        })
    return songs

@app.post("/api/user_recommendations")
def get_user_recommendations(payload: dict = Body(...)):
    ratings = payload.get("ratings", {})
    k = int(payload.get("k", 5))  

    candidate_vec = [ratings.get(song, 0) for song in song_cols]
    genero_predicho = classify_candidate(candidate_vec, k=k)
    recomendaciones = recommend_songs(candidate_vec, k=k, top_n=10, user_genre=None)

    return {"GeneroDetectado": genero_predicho, "Recomendaciones": recomendaciones}


@app.post("/api/recommend")
def recommend(candidate: Candidate):
    candidate_vec = [candidate.ratings.get(song, 0) for song in song_cols]
    genre = classify_candidate(candidate_vec)
    songs = recommend_songs(candidate_vec)
    return {"GeneroDetectado": genre, "Recomendaciones": songs}
