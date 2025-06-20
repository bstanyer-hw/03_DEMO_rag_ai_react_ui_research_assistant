import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";        // tailwind directives live here

// Mount React into the <div id="root"> inside index.html
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
