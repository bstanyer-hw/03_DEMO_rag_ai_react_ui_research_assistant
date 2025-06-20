import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import BotPage from "./pages/BotPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* default → news agent */}
        <Route path="/" element={<Navigate to="/news" replace />} />
        {/* dynamic slug for each specialised agent */}
        <Route path="/:botSlug" element={<BotPage />} />
      </Routes>
    </BrowserRouter>
  );
}
