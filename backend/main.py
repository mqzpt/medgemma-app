import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForImageTextToText, AutoProcessor

from inference import COMPARISON_PROMPT, RADIOLOGY_PROMPT, pil_from_upload, stream_inference

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
MODEL_ID = "google/medgemma-4b-it"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN is not set. Add it to backend/.env", file=sys.stderr)
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading {MODEL_ID} on {device}...")

    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        token=token,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    app.state.model = model
    app.state.processor = processor
    app.state.device = device
    print(f"Model loaded on {device}. Ready.")

    yield

    del app.state.model
    del app.state.processor


app = FastAPI(lifespan=lifespan)


@app.post("/api/interpret")
async def interpret(files: list[UploadFile] = File(...)):
    model = app.state.model
    processor = app.state.processor
    device = app.state.device

    # Read all file bytes now, while the request is still open.
    # UploadFile objects are closed before the streaming generator runs.
    file_data = [(f.filename or f"image_{i}", await f.read()) for i, f in enumerate(files)]

    async def event_stream():
        all_results: list[str] = []
        try:
            for i, (filename, data) in enumerate(file_data):
                yield f"data: {json.dumps({'type': 'image_start', 'index': i, 'name': filename})}\n\n"

                image = pil_from_upload(filename, data)
                result_text = ""

                async for token in stream_inference(model, processor, device, image, RADIOLOGY_PROMPT):
                    yield f"data: {json.dumps({'type': 'token', 'index': i, 'text': token})}\n\n"
                    result_text += token

                all_results.append(result_text)
                yield f"data: {json.dumps({'type': 'image_done', 'index': i})}\n\n"

            if len(file_data) > 1:
                yield f"data: {json.dumps({'type': 'comparison_start'})}\n\n"

                reports = "\n\n".join(
                    f"Image {i + 1} ({name}):\n{r}"
                    for i, ((name, _), r) in enumerate(zip(file_data, all_results))
                )
                prompt = COMPARISON_PROMPT.format(reports=reports)

                async for token in stream_inference(
                    model, processor, device, None, prompt, text_only=True
                ):
                    yield f"data: {json.dumps({'type': 'token', 'comparison': True, 'text': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Mount frontend last so API routes take priority
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
