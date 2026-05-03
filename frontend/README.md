# Frontend UI

Vite + React UI for the politician classifier. It calls the Flask backend at `/predict` and lets you choose the model (VGGFace2 or CASIA).

## Run locally

**Prerequisites:** Node.js

1. Install dependencies:
   `npm install`
2. (Optional) Create `.env.local` and set `VITE_API_URL` to your backend predict endpoint.
3. Start the dev server:
   `npm run dev`

## Config

- `VITE_API_URL`: full URL to the Flask `/predict` endpoint (default: `http://localhost:5000/predict`).
- You can override per session by opening `http://localhost:3000/?api=http://localhost:5000/predict`.
