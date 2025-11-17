# Face Blur — Image Face Anonymizer (Streamlit)

Simple image face anonymization app (blur / pixelate) built with OpenCV and Streamlit.
Includes interactive before/after slider and live detection tuning (scaleFactor, minNeighbors).

## Demo

Try the live demo: <https://your-deploy-link>

## Features

- Blur or pixelate faces in images
- Interactive before/after slider
- Live sliders for detection tuning (scaleFactor & minNeighbors)
- Local Haarcascade model (works offline)
- Download processed image

## Quick start (local)

1. Clone:

   ```bash
   git clone https://github.com/<yourusername>/face-blurr-app.git
   cd face-blurr-app

2. Create & activate virtual environment (optional):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

Run:
    ```bash
    streamlit run streamlit_app.py
  
Depoy to Streamlit Cloud:

1. Push your code to a GitHub repository.```
