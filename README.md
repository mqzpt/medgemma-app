# Local MedGemma — Chest X-Ray Interpreter

A simple, privacy-first web app for interpreting chest X-ray DICOM images using Google's MedGemma AI model, running entirely on your local machine.

## Why I built this

To assist in the diagnosis and risk measurement of lung nodules, without uploading sensitive medical images to any external service. This app lets you drop in the DICOM files from radiology exports and get structured AI interpretations instantly and all on-device.

The example images in this repo are real chest X-rays I got taken in October and November 2025. I can only upload the .jpg images as DICOM image metadata contains lots of privacy sensitive info, though .dcm is supported. 

## Features

- Drag and drop DICOM (`.dcm`) or image files (`.jpg`, `.png`)
- Streams the AI interpretation token by token as it generates
- Upload multiple images to get a comparative analysis across scans
- 100% local — no data leaves your machine

## Privacy

All inference runs on-device. Your images are never uploaded anywhere. The web server binds to `localhost` only.

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Add your Hugging Face token to .env (needed for one-time model download)
uvicorn main:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

The model (`google/medgemma-4b-it`, ~8GB) downloads on first run and is cached locally. You can remove the HF token from `.env` after that if you want.

## Requirements

- Python 3.10+
- Mac (Apple Silicon recommended for MPS acceleration), Linux, or Windows
- ~8GB disk space for model weights
- ~10GB RAM

## Disclaimer

Obviously for personal informational use only. AI output does not constitute medical advice and should not replace the judgment of a licensed radiologist or physician.

## Extra

Can take a while to run, and will depend very highly on your hardware. Local inference on my MBP 2020 M1 takes about 25 minutes per image. The model runs basically entirely on the neural engine / gpu cores on Apple Silicon, so your CPU is pretty free and you can just let this run in the background unless you're doing anything that requires heavy GPU workload. 

