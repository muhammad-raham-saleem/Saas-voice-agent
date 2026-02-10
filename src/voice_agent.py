import io
import os
import wave

from dotenv import load_dotenv

load_dotenv()

RECORD_SECONDS = 6
SAMPLE_RATE = 16000
CHANNELS = 1

EXIT_COMMANDS = {"bye", "goodbye", "exit", "quit", "stop"}


def _check_api_keys():
    """Validate that required API keys are present."""
    missing = []
    if not os.getenv("DEEPGRAM_API_KEY"):
        missing.append("DEEPGRAM_API_KEY")
    if not os.getenv("ELEVENLABS_API_KEY"):
        missing.append("ELEVENLABS_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        raise ValueError(f"❌ Missing API keys in .env: {', '.join(missing)}")
    print("✅ All API keys loaded")


def record_audio(duration: int = RECORD_SECONDS) -> bytes:
    """Record audio from the microphone and return WAV bytes.

    Args:
        duration: Recording duration in seconds.

    Returns:
        WAV file content as bytes.
    """
    try:
        import pyaudio
    except ImportError:
        raise RuntimeError("❌ PyAudio is not installed. Run: pip install pyaudio")

    audio = pyaudio.PyAudio()
    print(f"🎙️  Recording for {duration} seconds... Speak now!")

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=1024,
    )

    frames = []
    for _ in range(0, int(SAMPLE_RATE / 1024 * duration)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("✅ Recording complete")

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    return wav_buffer.getvalue()


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using Deepgram REST API.

    Args:
        audio_bytes: WAV file content.

    Returns:
        Transcribed text string.
    """
    import httpx

    response = httpx.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": "nova-3", "smart_format": "true", "language": "en"},
        headers={
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
            "Content-Type": "audio/wav",
        },
        content=audio_bytes,
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()


def speak_text(text: str):
    """Convert text to speech using ElevenLabs and play it.

    Args:
        text: The text to speak aloud.
    """
    from elevenlabs.client import ElevenLabs
    from elevenlabs.play import play

    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    audio = client.text_to_speech.convert(
        text=text,
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
    )

    play(audio)


def _has_pyaudio() -> bool:
    """Check if PyAudio is available."""
    try:
        import pyaudio
        return True
    except ImportError:
        return False


def _get_input(voice_mode: bool) -> str | None:
    """Get user input via microphone (voice mode) or keyboard.

    Returns:
        Transcribed/typed text, or None if no input detected.
    """
    if voice_mode:
        try:
            audio_bytes = record_audio()
            return transcribe_audio(audio_bytes)
        except Exception as e:
            print(f"⚠️  Recording error: {e}")
            print("💡 Type your question instead:")
            return input("You: ").strip() or None
    else:
        return input("You: ").strip() or None


def run_voice_agent():
    """Main loop: record → transcribe → RAG query → speak answer."""
    _check_api_keys()

    from src.rag_engine import RAGEngine

    voice_mode = _has_pyaudio()
    if voice_mode:
        mode_label = "🎙️  Voice mode (mic + speakers)"
    else:
        print("⚠️  PyAudio not installed — running in text-only mode")
        print("   To enable voice: sudo apt-get install portaudio19-dev && pip install pyaudio")
        mode_label = "⌨️  Text mode (type your questions)"

    print("\n" + "=" * 60)
    print("  🎧 TechFlow Voice Support Agent")
    print(f"  Mode: {mode_label}")
    print("  Type 'bye', 'goodbye', or 'exit' to end the session")
    print("=" * 60 + "\n")

    print("⏳ Initializing RAG engine...")
    engine = RAGEngine()
    print()

    greeting = "Hello! I'm the TechFlow support assistant. How can I help you today?"
    print(f"🤖 {greeting}")
    if voice_mode:
        try:
            speak_text(greeting)
        except Exception as e:
            print(f"⚠️  TTS unavailable ({e}), continuing without audio")

    while True:
        print()
        try:
            transcript = _get_input(voice_mode)
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not transcript:
            print("🔇 No input detected. Please try again.")
            continue

        print(f"🗣️  You said: \"{transcript}\"")

        if transcript.lower().strip(".!? ") in EXIT_COMMANDS:
            farewell = "Thank you for contacting TechFlow support. Have a great day!"
            print(f"🤖 {farewell}")
            if voice_mode:
                try:
                    speak_text(farewell)
                except Exception:
                    pass
            break

        print("🔍 Searching knowledge base...")
        try:
            result = engine.query(transcript)
        except Exception as e:
            error_msg = "I'm sorry, I encountered an error processing your question. Please try again."
            print(f"❌ Query error: {e}")
            print(f"🤖 {error_msg}")
            continue

        answer = result["answer"]
        print(f"🤖 {answer}")
        print(f"⏱️  ({result['response_time']}s | Sources: {', '.join(s['source'] for s in result['sources'])})")

        if voice_mode:
            try:
                speak_text(answer)
            except Exception as e:
                print(f"⚠️  TTS error: {e}")

    print("\n👋 Voice agent session ended.")


if __name__ == "__main__":
    run_voice_agent()
