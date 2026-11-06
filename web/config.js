// API base URL for the LKY chatbot backend.
// Production: the FastAPI service on Render (free tier, sleeps when idle).
// Local development: the FastAPI server runs on http://localhost:8000
// (uvicorn src.app:app --port 8000, from the project root).
// A ?api=... query parameter overrides it for one page load (dev only).
window.LKY_CONFIG = { API_BASE: "https://lky-chatbot.onrender.com" };
