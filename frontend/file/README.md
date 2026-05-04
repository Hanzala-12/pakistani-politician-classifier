# Pakistani Politician Classifier - Frontend

A sleek, responsive, glassmorphism UI for predicting Pakistani politicians from images.

## Features

- **Drag and Drop**: Upload images by dragging them onto the designated area or clicking to browse files.
- **Glassmorphism Design**: Modern UI with real-time blur and elegant gradient backgrounds.
- **Dynamic Previews**: View your selected image instantly.
- **Animated Confidence Bars**: See classification results via smoothly animated progress bars.
- **Error Handling**: Graceful messaging if a face is not detected or the server is down.

## Getting Started

1. You don't need a build system to run this frontend. It consists of vanilla HTML/CSS/JS files.
2. If your backend (port `5000`) and the UI run on different ports or domains, make sure your backend is configured to accept Cross-Origin (CORS) requests.

### Running the Frontend

Although you can double-click `index.html` to open it in a browser, handling API requests might trigger strict browser policies (CORS) if you run it locally off the `file://` protocol. 

It is better to serve the files using a simple local server:

1. Open a terminal.
2. Navigate to this `frontend` directory.
3. Start a basic HTTP server:
   - Python 3: `python -m http.server 8000`
   - Node.js (if `http-server` is installed): `npx http-server`
4. Open `http://localhost:8000` in your browser.

## Backend Integration

The frontend expects your provided REST API endpoint to be running locally at:
\`http://localhost:5000/predict\`

1. Ensure the endpoint accepts `multipart/form-data` with an image sent under the key `file`.
2. Ensure the endpoint returns JSON matching the spec:
   ```json
   {
     "predicted_class": "imran_khan",
     "confidence": 0.874,
     "top3": [
       {"class": "imran_khan", "confidence": 0.874},
       {"class": "nawaz_sharif", "confidence": 0.065},
       {"class": "shahbaz_sharif", "confidence": 0.032}
     ]
   }
   ```
3. If no face is detected or an error happens, return:
   ```json
   { "error": "No face detected" }
   ```

## Styling

- Font: [Inter via Google Fonts](https://fonts.google.com/specimen/Inter)
- Layout: Modern CSS Flexbox & CSS Grid, customized Custom Variables
- Design Pattern: Glassmorphism (`backdrop-filter`)

Enjoy classifying!
