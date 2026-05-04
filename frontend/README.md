<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/d6340e86-f364-41f6-b346-7f4fa7b4e84b

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Create a local env file from the example and set the backend URL:
   - Copy `.env.example` to `.env.local` (or create `.env`)
   - Edit `VITE_API_BASE_URL` to point to your backend, e.g. `VITE_API_BASE_URL="http://127.0.0.1:5000"`
3. (Optional) Set the `GEMINI_API_KEY` in `.env.local` if used by the app
4. Run the app:
   `npm run dev`

Notes:
- The frontend expects the backend to expose `POST /predict` (single) and `POST /predict/batch` (multipart form uploads). These are used for single-image and batch inference respectively.
- CORS must be enabled on the backend for `http://localhost:3000` in development. The project's backend currently enables CORS globally.
