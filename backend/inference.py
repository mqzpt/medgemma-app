import asyncio
from io import BytesIO
from threading import Thread

import numpy as np
import pydicom
import torch
from PIL import Image
from transformers import TextIteratorStreamer

RADIOLOGY_PROMPT = """\
You are an expert radiologist interpreting a chest X-ray. Provide a structured report covering:

1. **Technical Quality** — positioning, exposure, rotation, inspiratory effort
2. **Lung Fields** — any opacities, consolidation, infiltrates, nodules, masses, air trapping
3. **Pleural Spaces** — effusion, pneumothorax, thickening
4. **Cardiac Silhouette** — size, shape, borders
5. **Mediastinum & Hila** — width, contour, lymphadenopathy
6. **Bones & Soft Tissues** — ribs, spine, clavicles, visible soft tissue
7. **Impression** — concise summary of key findings
8. **Recommendations** — any follow-up or further imaging suggested

Be precise and thorough."""

COMPARISON_PROMPT = """\
You are a radiologist comparing multiple chest X-ray studies for the same patient.

Based on the individual interpretations below, provide:
1. **Interval changes** — what has changed across the studies?
2. **Stable findings** — what remains consistent?
3. **New findings** — anything present in later images not in earlier ones?
4. **Resolved findings** — anything in earlier images no longer seen?
5. **Overall impression** — is the condition improving, stable, or worsening?

Individual interpretations:
{reports}"""


def dcm_to_pil(data: bytes) -> Image.Image:
    ds = pydicom.dcmread(BytesIO(data))
    pixels = ds.pixel_array.astype(float)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixels = pixels * slope + intercept

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    wc = float(wc[0] if hasattr(wc, "__iter__") else wc) if wc is not None else -600
    ww = float(ww[0] if hasattr(ww, "__iter__") else ww) if ww is not None else 1500

    lo, hi = wc - ww / 2, wc + ww / 2
    pixels = np.clip(pixels, lo, hi)
    pixels = ((pixels - lo) / (hi - lo) * 255).astype(np.uint8)

    img = Image.fromarray(pixels).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    return img


def pil_from_upload(filename: str, data: bytes) -> Image.Image:
    if filename.lower().endswith(".dcm"):
        return dcm_to_pil(data)
    return Image.open(BytesIO(data)).convert("RGB")


async def stream_inference(model, processor, device, image, prompt, text_only=False):
    """Async generator that yields tokens as they are produced by the model."""
    if text_only:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tokenizer = getattr(processor, "tokenizer", processor)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    # Thread 1: model.generate feeds tokens into streamer
    gen_thread = Thread(
        target=model.generate,
        kwargs={**inputs, "max_new_tokens": 1000, "do_sample": False, "streamer": streamer},
        daemon=True,
    )

    # Thread 2: drains streamer into asyncio queue so we don't block the event loop
    def drain():
        for token in streamer:
            loop.call_soon_threadsafe(queue.put_nowait, token)
        loop.call_soon_threadsafe(queue.put_nowait, None)

    drain_thread = Thread(target=drain, daemon=True)

    gen_thread.start()
    drain_thread.start()

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token

    gen_thread.join(timeout=10)
    drain_thread.join(timeout=10)
