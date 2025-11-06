// frontend/src/Login.jsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();

    if (!name || !email) {
      alert("Por favor, completa todos los campos.");
      return;
    }

    // Simula guardar usuario (sin backend)
    localStorage.setItem("user", JSON.stringify({ name, email }));

    // Redirige a la página principal
    navigate("/home");
  };

  return (
    <div style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      height: "100vh",
      width: "100%",
      background: "linear-gradient(135deg, #1DB954, #191414)",
      color: "white"
    }}>
      <form
        onSubmit={handleLogin}
        style={{
          background: "rgba(0,0,0,0.7)",
          padding: "40px",
          borderRadius: "12px",
          textAlign: "center",
          width: "300px"
        }}
      >
        <h2>🎧 Iniciar Sesión</h2>

        <input
          type="text"
          placeholder="Nombre"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{
            width: "100%",
            padding: "10px",
            margin: "10px 0",
            borderRadius: "6px",
            border: "none"
          }}
        />

        <input
          type="email"
          placeholder="Correo electrónico"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={{
            width: "100%",
            padding: "10px",
            margin: "10px 0",
            borderRadius: "6px",
            border: "none"
          }}
        />

        <button
          type="submit"
          style={{
            backgroundColor: "#1DB954",
            color: "white",
            border: "none",
            borderRadius: "6px",
            padding: "10px 20px",
            marginTop: "10px",
            cursor: "pointer",
            width: "100%",
            fontWeight: "bold"
          }}
        >
          Entrar 🚀
        </button>
      </form>
    </div>
  );
}
