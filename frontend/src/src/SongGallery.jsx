// frontend/src/SongGallery.jsx
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export default function SongGallery() {
  const [songs, setSongs] = useState([]);
  const [ratings, setRatings] = useState({});
  const navigate = useNavigate();

  useEffect(() => {
    axios
      .get("http://localhost:8000/api/songs")
      .then((res) => setSongs(res.data))
      .catch((err) => console.error("Error al obtener canciones:", err));
  }, []);

  const handleRate = (song, rating) => {
    setRatings((prev) => ({ ...prev, [song]: rating }));
  };

const sendRatings = async () => {
  try {
    // Guardar ratings actualizados en localStorage
    localStorage.setItem("ratings", JSON.stringify(ratings));

    // Llamada al backend
    const res = await axios.post("http://localhost:8000/api/recommend", {
      ratings,
    });

    const genero = res.data.GeneroDetectado;
    localStorage.setItem("generoFavorito", genero);
    localStorage.setItem("recomendaciones", JSON.stringify(res.data.Recomendaciones));

    alert("Género detectado: " + genero);

    navigate("/perfil");
  } catch (error) {
    console.error("Error al enviar calificaciones:", error);
  }
};


  const logout = () => {
    localStorage.removeItem("user");
    window.location.href = "/";
  };

  return (
    <div
      style={{
        padding: "30px",
        background: "linear-gradient(180deg, #121212, #1e1e1e)",
        minHeight: "100vh",
        color: "#fff",
        fontFamily: "Arial, sans-serif",
      }}
    >
      {/* --- BOTONES SUPERIORES --- */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: "20px",
        }}
      >
        <button
          onClick={logout}
          style={{
            background: "#e63946",
            color: "white",
            border: "none",
            borderRadius: "8px",
            padding: "10px 18px",
            cursor: "pointer",
            fontSize: "15px",
            fontWeight: "bold",
            boxShadow: "0 4px 10px rgba(0,0,0,0.3)",
            transition: "background 0.3s, transform 0.2s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#c72f3b")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "#e63946")}
        >
          ✖ Cerrar sesión
        </button>

        <button
          onClick={() => navigate("/perfil")}
          style={{
            background: "#1DB954",
            color: "white",
            border: "none",
            borderRadius: "8px",
            padding: "10px 18px",
            cursor: "pointer",
            fontSize: "15px",
            fontWeight: "bold",
            boxShadow: "0 4px 10px rgba(0,0,0,0.3)",
            transition: "background 0.3s, transform 0.2s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#17a74b")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "#1DB954")}
        >
          👤 Ver perfil
        </button>
      </div>

      <h1 style={{ textAlign: "center", marginBottom: "30px", color: "#1DB954" }}>
        Califica las canciones
      </h1>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
          gap: "25px",
          justifyItems: "center",
        }}
      >
        {songs.map((song, i) => (
          <div
            key={i}
            style={{
              borderRadius: "12px",
              padding: "15px",
              textAlign: "center",
              background:
                "linear-gradient(180deg, rgba(40,40,40,0.95), rgba(20,20,20,0.95))",
              boxShadow: "0 4px 15px rgba(0,0,0,0.4)",
              width: "100%",
              maxWidth: "260px",
              transition: "transform 0.2s, box-shadow 0.2s",
              cursor: "pointer",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "scale(1.03)";
              e.currentTarget.style.boxShadow = "0 6px 18px rgba(0,0,0,0.6)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "scale(1)";
              e.currentTarget.style.boxShadow = "0 4px 15px rgba(0,0,0,0.4)";
            }}
          >
            <img
              src={song.Portada || song.Imagen || "https://via.placeholder.com/250?text=No+Image"}
              alt={song.Nombre}
              style={{
                width: "100%",
                borderRadius: "10px",
                marginBottom: "10px",
              }}
            />

            <h3 style={{ margin: "5px 0", color: "#fff" }}>{song.Nombre}</h3>
            <p style={{ fontWeight: "bold", margin: "2px 0", color: "#ddd" }}>
              🎤 {song.Artista}
            </p>
            <p style={{ margin: "2px 0", color: "#aaa" }}>💿 {song.Album}</p>
            <p style={{ fontStyle: "italic", margin: "2px 0", color: "#1DB954" }}>
              🎶 {song.Genero}
            </p>

            <a
              href={song.URL}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: "inline-block",
                margin: "8px 0",
                color: "#1DB954",
                textDecoration: "none",
                fontWeight: "bold",
                fontSize: "14px",
              }}
            >
              🔗 Escuchar en Spotify
            </a>

            <div style={{ marginTop: "8px" }}>
              {[1, 2, 3, 4, 5].map((star) => (
                <span
                  key={star}
                  onClick={() => handleRate(song.Nombre, star)}
                  style={{
                    cursor: "pointer",
                    color:
                      (ratings[song.Nombre] || 0) >= star ? "#FFD700" : "#555",
                    fontSize: "22px",
                    transition: "color 0.2s",
                  }}
                >
                  ★
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div style={{ textAlign: "center" }}>
        <button
          onClick={sendRatings}
          style={{
            marginTop: "40px",
            padding: "12px 25px",
            fontSize: "18px",
            cursor: "pointer",
            backgroundColor: "#1DB954",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            boxShadow: "0 4px 10px rgba(0,0,0,0.3)",
            transition: "background 0.3s, transform 0.2s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#17a74b")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "#1DB954")}
        >
          Enviar Calificaciones
        </button>
      </div>
    </div>
  );
}
