
import os
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv(override=True)

api_key = os.getenv("ELEVENLABS_API_KEY")

if not api_key:
    raise SystemExit(
        "ERROR: ELEVENLABS_API_KEY is missing from your .env file."
    )

client = ElevenLabs(api_key=api_key)

try:
    print("Checking voices available to your account...")
    response = client.voices.search()

    if not response.voices:
        raise SystemExit(
            "No voices were returned. Check your account and API permissions."
        )

    voice = response.voices[0]
    print(f"Using voice: {voice.name}")

    print("Generating test speech...")
    audio = client.text_to_speech.convert(
        text="Hello! Welcome to your AI mock interview coach. "
             "Let's begin your interview.",
        voice_id=voice.voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    output_path = Path("test_voice.mp3")

    with output_path.open("wb") as audio_file:
        for chunk in audio:
            audio_file.write(chunk)

    print(f"SUCCESS! Audio saved to: {output_path.resolve()}")
    print("Opening the generated audio...")

    os.system(f"open {str(output_path)!r}")

except SystemExit:
    raise
except Exception as error:
    print(f"VOICE TEST FAILED: {type(error).__name__}: {error}")
    print("Check your API key, account access, and available credits.")
