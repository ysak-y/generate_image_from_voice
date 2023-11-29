from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from wave import Wave_write
import base64
from google.cloud import speech
from diffusers import LCMScheduler, StableDiffusionXLPipeline
import torch
from google.cloud import translate_v2

app = FastAPI()

model_id = "segmind/SSD-1B"
lcm_lora_id = "latent-consistency/lcm-lora-ssd-1b"

pipe = StableDiffusionXLPipeline.from_pretrained(model_id, variant="fp16")

pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device="mps", dtype=torch.float16)

app.mount("/static", StaticFiles(directory="./static"), name="static") # Mount folder to expose generated images

@app.post("/")
async def post_root(request: Request):
    writer = Wave_write("test.wav")
    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(44100)
    frames = []
    chunk_total = 0

    async for chunk in request.stream():
        chunk_total += len(chunk)
        frames.extend(chunk)

    buffer = base64.urlsafe_b64decode(bytearray(frames))
    writer.writeframes(bytearray(buffer))

    # Instantiates a client
    client = speech.SpeechClient()

    with open("test.wav", "rb") as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="ja-JP",
        )

        # Detects speech in the audio file
        response = client.recognize(config=config, audio=audio)
        transcript = response.results[0].alternatives[0].transcript
        print("Recognized: ", transcript)

        translate_client = translate_v2.Client()
        result = translate_client.translate(transcript, target_language="en")
        translated = result["translatedText"]
        print("Translated: ", translated)

        neg_prompt = "ugly, blurry, prro quality"
        images = pipe(
            prompt=translated,
            negative_prompt=neg_prompt,
            num_inference_steps=2,
            guidance_scale=1,
        ).images[0]

        images.save("static/img1.png")
        return { "message": "ok" }
