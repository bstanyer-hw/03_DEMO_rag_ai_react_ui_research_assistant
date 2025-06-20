// src/components/Message.jsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/* ------------------------------------------------------------------ */
/*  ChatGPT-like visual tokens                                        */
/* ------------------------------------------------------------------ */
const USER_BUBBLE =
  "bg-[#d1e8ff] text-gray-900 ml-auto rounded-lg px-4 py-2 max-w-[80%]";  // blue-tint
const AI_BLOCK =
  "bg-[#f7f7f8] text-gray-900 rounded-lg border border-gray-200 shadow-sm px-6 py-4 w-full max-w-none";

/* Markdown → Tailwind component map mimicking ChatGPT light theme */
const mdComponents = {
  /* Larger headings with bold weight */
  h1: ({node, ...props}) => <h2 className="font-bold text-2xl mt-4 mb-2" {...props} />,
  h2: ({node, ...props}) => <h3 className="font-semibold text-xl mt-4 mb-2" {...props} />,
  h3: ({node, ...props}) => <h4 className="font-semibold text-lg mt-3 mb-1" {...props} />,

  /* Paragraph */
  p:  ({node, ...props}) => <p className="mb-3 leading-[1.6]" {...props} />,

  /* Strong / em */
  strong: ({node, ...props}) => <strong className="font-semibold text-gray-900" {...props} />,
  em:     ({node, ...props}) => <em className="italic" {...props} />,

  /* Lists */
  ul: ({node, ...props}) => <ul className="list-disc ml-6 space-y-1 mb-3" {...props} />,
  ol: ({node, ...props}) => <ol className="list-decimal ml-6 space-y-1 mb-3" {...props} />,
  li: ({node, ...props}) => <li {...props} />,

  /* Links */
  a: ({node, ...props}) => (
    <a
      className="text-blue-600 hover:underline"
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    />
  ),

  /* Code */
  code: ({node, inline, ...props}) =>
    inline ? (
      <code className="font-mono bg-gray-200 px-1 rounded" {...props} />
    ) : (
      <pre className="font-mono bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-3">
        <code {...props} />
      </pre>
    ),

  /* Blockquote */
  blockquote: ({node, ...props}) => (
    <blockquote className="border-l-4 border-gray-300 pl-4 italic text-gray-700 mb-3" {...props} />
  ),

  hr: () => <hr className="my-5 border-gray-300" />,
};

export default function Message({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} my-2`}>
      <div className={isUser ? USER_BUBBLE : AI_BLOCK}>
        {isUser ? (
          message.content
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
            {message.content}
          </ReactMarkdown>
        )}
      </div>
    </div>
  );
}


