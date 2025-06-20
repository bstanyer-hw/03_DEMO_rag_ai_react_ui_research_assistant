// src/components/ChatBox.jsx
import { useState, useRef, useEffect } from "react";
import Message from "./Message";

export default function ChatBox({ botSlug }) {
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(false);
  const inputRef  = useRef(null);
  const scrollRef = useRef(null);          // chat container

  /* --- keep the view pinned to the latest message --- */
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    const question = inputRef.current.value.trim();
    if (!question || pending) return;

    /* 1️⃣  push user message + empty assistant placeholder */
    setMessages((m) => [
      ...m,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);
    inputRef.current.value = "";
    setPending(true);

    /* --- streaming call --------------------------------------------------- */
    const res = await fetch(import.meta.env.VITE_API_BASE + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history_summary: "", bot: botSlug }),
    });

    if (!res.ok || !res.body) {
      setPending(false);
      alert("Error contacting server");
      return;
    }

    const reader = res.body.getReader();
    let accumulated = "";
    const textDecoder = new TextDecoder();

    /* 2️⃣  read & grow the placeholder instead of pushing new bubbles */
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      accumulated += textDecoder.decode(value);

      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { role: "assistant", content: accumulated };
        return copy;
      });
    }
    setPending(false);
  };

  return (
    <>
      <div
        ref={scrollRef}
        className="border rounded-lg p-4 space-y-4 h-[70vh] overflow-y-auto"
      >
        {messages.map((msg, i) => (
          <Message key={i} message={msg} />
        ))}
      </div>

      <div className="mt-4 flex gap-2">
        <input
          ref={inputRef}
          type="text"
          placeholder="Ask something…"
          className="flex-1 border rounded-lg px-3 py-2"
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          disabled={pending}
          onClick={sendMessage}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </>
  );
}

