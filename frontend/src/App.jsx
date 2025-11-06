// frontend/src/App.jsx
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Login from "./src/Login";
import SongGallery from "./src/SongGallery";
import UserProfile from "./src/UserProfile";

function ProtectedRoute({ children }) {
  const user = localStorage.getItem("user");
  return user ? children : <Navigate to="/" />;
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route
          path="/home"
          element={
            <ProtectedRoute>
              <SongGallery />
            </ProtectedRoute>
          }
        />
        <Route
          path="/perfil"
          element={
            <ProtectedRoute>
              <UserProfile />
            </ProtectedRoute>
          }
        />
      </Routes>
    </Router>
  );
}
