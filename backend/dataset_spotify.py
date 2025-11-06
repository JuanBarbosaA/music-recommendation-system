import requests  # Para hacer solicitudes HTTP a la API de Spotify
import base64    # Para codificar credenciales en base64
import random    # Para generar valores aleatorios
import csv       # Para crear y escribir archivos CSV

# ================================
# 🔑 AUTENTICACIÓN SPOTIFY API
# ================================
# Credenciales de la aplicación Spotify
client_id = "707d8850cf45472c82251fccfc9aeefb"
client_secret = "b3007632f7c7414cab834d0d443f7d7d"

# Combina ID y secreto en un solo string y lo codifica en base64
auth_string = f"{client_id}:{client_secret}"
b64_auth_string = base64.b64encode(auth_string.encode()).decode()

# URL y headers para obtener el token de acceso
url_token = "https://accounts.spotify.com/api/token"
headers_token = {"Authorization": f"Basic {b64_auth_string}"}
data_token = {"grant_type": "client_credentials"}  # Tipo de autorización

# Solicitud POST a Spotify para obtener el access token
response = requests.post(url_token, headers=headers_token, data=data_token)
access_token = response.json()["access_token"]  # Extrae el token del JSON recibido

# Headers para futuras solicitudes a la API con el token
headers = {"Authorization": f"Bearer {access_token}"}
tracks = []  # Lista para almacenar información de canciones

# ================================
# 🎵 GÉNEROS Y RECOLECCIÓN DE CANCIONES
# ================================
generos = ["rock", "pop", "clasica", "reggae", "salsa", "vallenato"]  # Géneros a buscar
limite_por_genero = 40  # Número máximo de canciones por género

# Bucle por cada género para buscar canciones en Spotify
for genero in generos:
    url_search = f"https://api.spotify.com/v1/search?q={genero}&type=track&limit={limite_por_genero}"
    response = requests.get(url_search, headers=headers)
    data = response.json()  # Convierte la respuesta a JSON

    # Extrae información relevante de cada canción
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
            "url_spotify": url_spotify,
        }
        tracks.append(track_info)  # Añade la canción a la lista

# ================================
# 👥 GENERACIÓN DE USUARIOS Y CALIFICACIONES
# ================================
num_usuarios = 3000  # Número de usuarios simulados
usuarios = []  # Lista para almacenar información de usuarios
track_genero = [t["genero"] for t in tracks]  # Lista de géneros de todas las canciones

# Genera usuarios aleatorios
for user_id in range(1, num_usuarios + 1):
    edad = random.randint(15, 60)
    genero_usuario = random.choice(["M", "F"])
    region = random.choice(["Caribe", "Llanos", "Bogotá", "Santander"])
    genero_fav = random.choice(generos)
    clase_usuario = f"Amante del {genero_fav.capitalize()}"

    # Genera calificaciones de canciones según el género favorito del usuario
    calificaciones = []
    for g in track_genero:
        if g == genero_fav:
            # Más probabilidades de calificaciones altas para su género favorito
            calificaciones.append(random.choices([3, 4, 5], weights=[1, 3, 6])[0])
        else:
            # Calificaciones más bajas para otros géneros
            calificaciones.append(random.choices([0, 1, 2, 3, 4], weights=[3, 3, 2, 1, 1])[0])

    usuario = [user_id, edad, genero_usuario, region, genero_fav.capitalize(), clase_usuario] + calificaciones
    usuarios.append(usuario)  # Añade el usuario a la lista

# ================================
# ⚠️ AÑADIR RUIDO (30% USUARIOS CAMBIADOS)
# ================================
random.shuffle(usuarios)  # Mezcla aleatoriamente los usuarios
porcentaje_ruido = 0.3
num_ruido = int(num_usuarios * porcentaje_ruido)

# Cambia el género favorito de algunos usuarios para simular ruido
for i in range(num_ruido):
    actual_genero = usuarios[i][4]
    posibles_generos = [g.capitalize() for g in generos if g.capitalize() != actual_genero]
    nuevo_genero = random.choice(posibles_generos)
    usuarios[i][4] = nuevo_genero
    usuarios[i][5] = f"Amante del {nuevo_genero}"

print(f"⚙️ Se modificaron {num_ruido} usuarios ({porcentaje_ruido*100:.0f}%) para añadir ruido.\n")

# ================================
# 💾 GUARDAR CSVs
# ================================
# Header para el CSV de usuarios con calificaciones
header = ["UserID", "Edad", "Genero", "Region", "GeneroFav", "ClaseUsuario"] + [t["nombre"] for t in tracks]

# Guardar dataset de usuarios y sus calificaciones
with open("dataset_canciones_spotify.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(usuarios)

# Guardar información de las canciones
with open("info_canciones_spotify.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Nombre", "Artista", "Álbum", "Género", "SpotifyID", "Imagen", "URL_Spotify"])
    for t in tracks:
        writer.writerow([t["nombre"], t["artista"], t["album"], t["genero"], t["id_spotify"], t["imagen"], t["url_spotify"]])

print("✅ Dataset generado con ruido controlado (70% limpio, 30% modificado)")
print("✅ Archivos guardados: dataset_canciones_spotify.csv y info_canciones_spotify.csv")
