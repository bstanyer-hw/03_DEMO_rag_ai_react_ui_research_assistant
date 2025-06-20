import { useParams } from "react-router-dom";
import Header from "../components/Header";
import ChatBox from "../components/ChatBox";

export default function BotPage() {
  const { botSlug } = useParams();            // e.g. “news” | “earnings”
  return (
    <div className="flex flex-col h-screen">
      <Header />
      <main className="flex-1 max-w-4xl mx-auto w-full p-4">
        {/* description could be fetched by slug; hard-code for demo */}
        <p className="text-center mb-4 text-sm text-gray-600">
          {botSlug === "news"
            ? "Summarises and analyses financial & economic news articles."
            : `Specialised RAG agent: ${botSlug}`}
        </p>
        <ChatBox botSlug={botSlug} />
      </main>
    </div>
  );
}
