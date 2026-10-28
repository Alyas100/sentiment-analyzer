# Local CPU React dashboard

This dashboard is the local version of the project. It calls the local FastAPI
server, which loads `models/modernbert-sentiment` and performs CPU inference.
It does not connect to Hugging Face or the hosted Space.

## Local development

```bash
npm install
npm run dev
```

Start the API from the project root first:

```bash
make local-api
```

Then open the Vite URL, normally `http://localhost:5173`.

The Hugging Face-only Gradio app is kept separately in `../hf_space/app.py`.
