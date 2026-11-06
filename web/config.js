// API base URL for the LKY chatbot backend.
// Local development: the FastAPI server runs on http://localhost:8000
// (uvicorn src.app:app --port 8000, from the project root).
// After deploy, change this to the Render service URL, e.g.
// "https://lky-chatbot.onrender.com". The URL is not a secret.
// A ?api=... query parameter overrides it for one page load (dev only).
window.LKY_CONFIG = { API_BASE: "http://localhost:8000" };
