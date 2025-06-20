// frontend/src/components/Header.jsx
import logo from "/company_logo.png";

export default function Header() {
  return (
    <header className="border-b flex flex-col items-center py-4">
      <img src={logo} alt="Company logo" className="w-48 mb-2" />
      <h1 className="text-2xl font-semibold">AI RAG Chatbot</h1>
    </header>
  );
}

