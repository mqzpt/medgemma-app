# MedGemma App

Local, privacy-first chest X-ray interpreter. All inference runs on-device (Apple MPS). No patient data ever leaves the machine.

## Stack

- **Backend** — FastAPI + uvicorn, Python 3.13
- **Model** — `google/medgemma-4b-it` via Hugging Face `transformers`, loaded once on startup
- **Device** — Apple MPS (M1); falls back to CPU
- **Frontend** — Plain HTML/CSS/JS, no build step

## Project layout

```
medgemma-app/
├── backend/
│   ├── main.py        FastAPI app — lifespan model loading, /api/interpret SSE endpoint
│   ├── inference.py   DICOM → PIL conversion, async streaming inference, prompts
│   ├── .env           HF_TOKEN (gitignored, never commit)
│   ├── .env.example
│   └── requirements.txt
└── frontend/
    ├── index.html     Upload UI
    ├── style.css      Dark medical theme
    └── app.js         Fetch + ReadableStream SSE parser
```

## Running

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # add HF_TOKEN
uvicorn main:app --port 8000
# open http://localhost:8000
```

The model (~8GB) downloads on first run and caches at `~/.cache/huggingface/`. Subsequent startups load from cache in ~2 seconds.

## API

### `POST /api/interpret`

Accepts `multipart/form-data` with one or more files (`.dcm`, `.jpg`, `.png`).

Returns a `text/event-stream` of newline-delimited JSON events:

| Event type | Fields | Description |
|---|---|---|
| `image_start` | `index`, `name` | New image beginning |
| `token` | `index`, `text` | Streamed token for image `index` |
| `image_done` | `index` | Image interpretation complete |
| `comparison_start` | — | Comparative analysis starting (>1 image) |
| `token` | `comparison: true`, `text` | Streamed comparison token |
| `done` | — | All done |
| `error` | `message` | Something went wrong |

## Privacy

- DICOM files are processed entirely in memory on-device
- The web server binds to `127.0.0.1` only
- HF token is only used for the one-time model download
- No analytics, no logging of image content

## Key implementation notes

- `UploadFile` bytes must be read eagerly in the route handler before the `StreamingResponse` generator starts — files are closed by then.
- `TextIteratorStreamer` runs model generation in a daemon thread; a second thread drains tokens into an `asyncio.Queue` to avoid blocking the event loop.
- DICOM windowing defaults to lung window (WC −600 / WW 1500) when DICOM tags are absent.
