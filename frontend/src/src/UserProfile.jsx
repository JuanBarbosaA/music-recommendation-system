import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export default function UserProfile() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [genero, setGenero] = useState("");
  const [numSongs, setNumSongs] = useState(5);
  const [songs, setSongs] = useState([]);
  const [ratings, setRatings] = useState(JSON.parse(localStorage.getItem("ratings")) || {});
  const [kValue, setKValue] = useState(5);

  // 🔹 Cargar usuario al montar el componente
  useEffect(() => {
    const storedUser = JSON.parse(localStorage.getItem("user"));
    if (storedUser) {
      setUser(storedUser);
    }
  }, []);

  // 🔹 Traer recomendaciones cada vez que cambian las calificaciones
  useEffect(() => {
    const fetchRecommendations = async () => {
      try {
        if (Object.keys(ratings).length === 0) return;

        const res = await axios.post("http://localhost:8000/api/user_recommendations", {
          ratings,
          k: kValue,
        });

        setGenero(res.data.GeneroDetectado);
        setSongs(res.data.Recomendaciones);

        localStorage.setItem("generoFavorito", res.data.GeneroDetectado);
        localStorage.setItem("recomendaciones", JSON.stringify(res.data.Recomendaciones));
      } catch (err) {
        console.error("Error al obtener recomendaciones:", err);
      }
    };

    fetchRecommendations();
}, [ratings, kValue]);

  const handleChange = (e) => setNumSongs(Number(e.target.value));

  // 🔹 Actualizar calificación de una canción y disparar nueva recomendación
  const handleRatingChange = (song, score) => {
    const newRatings = { ...ratings, [song]: score };
    setRatings(newRatings);
    localStorage.setItem("ratings", JSON.stringify(newRatings));
  };

  return (
    <div
      style={{
        padding: "40px",
        background: "linear-gradient(180deg, #121212, #1e1e1e)",
        minHeight: "100vh",
        color: "#fff",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <button
        onClick={() => navigate("/home")}
        style={{
          background: "#1DB954",
          color: "white",
          border: "none",
          borderRadius: "8px",
          padding: "10px 20px",
          cursor: "pointer",
          marginBottom: "30px",
          fontSize: "15px",
          fontWeight: "bold",
          boxShadow: "0 4px 10px rgba(0,0,0,0.3)",
        }}
      >
        ⬅ Volver al inicio
      </button>

      <div
        style={{
          background: "linear-gradient(180deg, rgba(40,40,40,0.95), rgba(20,20,20,0.95))",
          borderRadius: "12px",
          padding: "30px",
          boxShadow: "0 6px 20px rgba(0,0,0,0.5)",
          maxWidth: "800px",
          margin: "0 auto 40px",
        }}
      >
        <h1 style={{ textAlign: "center", color: "#1DB954", marginBottom: "20px" }}>
          Perfil del Usuario
        </h1>

        {user ? (
          <>
            <p><strong>Nombre:</strong> {user.name}</p>
            <p><strong>Correo:</strong> {user.email}</p>
            <p>
              <strong>Género favorito:</strong>{" "}
              <span style={{ color: "#1DB954" }}>{genero || "Sin determinar"}</span>
            </p>


            <div style={{ marginTop: "25px" }}>
  <label>Valor de K (vecinos):</label>
  <input
    type="number"
    min="1"
    max="50"
    value={kValue}
    onChange={(e) => setKValue(Number(e.target.value))}
    style={{
      marginLeft: "10px",
      width: "70px",
      padding: "6px",
      borderRadius: "6px",
      border: "1px solid #1DB954",
      background: "#1b1b1b",
      color: "#fff",
      fontSize: "16px",
      textAlign: "center",
    }}
  />
</div>


            {songs.length > 0 && (
              <div style={{ marginTop: "25px" }}>
                <label>Mostrar número de canciones recomendadas:</label>
                <input
                  type="number"
                  min="1"
                  max={songs.length}
                  value={numSongs}
                  onChange={handleChange}
                  style={{
                    marginLeft: "10px",
                    width: "70px",
                    padding: "6px",
                    borderRadius: "6px",
                    border: "1px solid #1DB954",
                    background: "#1b1b1b",
                    color: "#fff",
                    fontSize: "16px",
                    textAlign: "center",
                  }}
                />
              </div>
            )}
          </>
        ) : (
          <p>No hay datos del usuario.</p>
        )}
      </div>

      {songs.length > 0 ? (
        <>
          <h2 style={{ textAlign: "center", color: "#1DB954" }}>🎶 Canciones recomendadas</h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
              gap: "25px",
            }}
          >
            {songs.slice(0, numSongs).map((song, i) => (
              <div
                key={i}
                style={{
                  background: "linear-gradient(180deg, rgba(40,40,40,0.95), rgba(20,20,20,0.95))",
                  borderRadius: "12px",
                  padding: "15px",
                  textAlign: "center",
                  boxShadow: "0 4px 15px rgba(0,0,0,0.4)",
                  transition: "transform 0.2s, box-shadow 0.2s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "scale(1.05)";
                  e.currentTarget.style.boxShadow = "0 6px 20px rgba(0,0,0,0.6)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                  e.currentTarget.style.boxShadow = "0 4px 15px rgba(0,0,0,0.4)";
                }}
              >
                {song.Imagen && (
                  <img
                    src={song.Imagen}
                    alt={song.Nombre}
                    style={{ width: "100%", borderRadius: "8px", marginBottom: "10px" }}
                  />
                )}
                <h3>{song.Nombre}</h3>
                <p>🎤 {song.Artista}</p>
                <p>💿 {song.Album}</p>
                <p style={{ color: "#1DB954" }}>🎶 {song.Genero}</p>
                <a
                  href={song.URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: "#1DB954", textDecoration: "none", fontWeight: "bold" }}
                >
                  🔗 Escuchar en Spotify
                </a>

                {/* Selector de calificación */}
                <div style={{ marginTop: "8px" }}>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <span
                      key={star}
                      onClick={() => handleRatingChange(song.Nombre, star)}
                      style={{
                        cursor: "pointer",
                        color: (ratings[song.Nombre] || 0) >= star ? "#FFD700" : "#555",
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
        </>
      ) : (
        <p style={{ textAlign: "center", marginTop: "40px" }}>
          No hay recomendaciones disponibles todavía.
        </p>
      )}
    </div>
  );
}
