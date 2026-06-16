# ElevenLabs Local Cloned Voice Demo

This project is a local Gradio UI for testing cloned-voice generation with ElevenLabs.

Supported workflows:

1. Text input → cloned voice output
2. Recorded microphone/uploaded audio clip → transcribe → cloned voice output
3. Continuous local microphone mode → auto-detect speech end → cloned voice output sentence by sentence
4. Optional merged audio output from generated chunks

The recommended migration path is:

```text
Develop/test on Linux first → confirm local mic and ElevenLabs API → migrate to Windows → use Windows local mic
```

---

## 1. Project files

Suggested folder structure:

```text
elevenlabs_voice_changer_demo/
├── app_continuous.py
├── requirements.txt
├── .env
└── README.md
```

Create `requirements.txt`:

```txt
gradio>=4.44.0
requests>=2.31.0
python-dotenv>=1.0.0
sounddevice>=0.4.6
webrtcvad-wheels>=2.0.14
numpy>=1.26.0
```

Create `.env`:

```env
ELEVENLABS_API_KEY=your_api_key_here
```

Do not commit or share `.env`.

---

## 2. Linux setup

### 2.1 Install system audio dependency

`sounddevice` needs PortAudio.

```bash
sudo apt update
sudo apt install -y portaudio19-dev libportaudio2 libportaudiocpp0
```

Optional audio tools for checking devices:

```bash
sudo apt install -y alsa-utils pulseaudio-utils
```

### 2.2 Create Python virtual environment

```bash
cd /path/to/elevenlabs_voice_changer_demo
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Check microphone devices

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

If this prints a device list, PortAudio and `sounddevice` are working.

If you get:

```text
OSError: PortAudio library not found
```

run:

```bash
sudo apt install -y portaudio19-dev libportaudio2 libportaudiocpp0
pip uninstall -y sounddevice
pip install sounddevice
```

### 2.4 Run the app

```bash
python app_continuous.py
```

Open:

```text
http://127.0.0.1:7860
```

To open from another machine on the same network:

```bash
hostname -I
```

Then open:

```text
http://LINUX_MACHINE_IP:7860
```

If blocked by firewall:

```bash
sudo ufw allow 7860/tcp
```

Important: in the current local-microphone version, the microphone used is the microphone connected to the machine running Python, not the remote browser machine.

---

## 3. Windows setup

### 3.1 Install Python

Install Python 3.10, 3.11, or 3.12 from the official Python website.

During installation, tick:

```text
Add python.exe to PATH
```

### 3.2 Create virtual environment

Open PowerShell:

```powershell
cd C:\path\to\elevenlabs_voice_changer_demo
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 Create `.env`

Create a file called `.env` in the project folder:

```env
ELEVENLABS_API_KEY=your_api_key_here
```

### 3.4 Check microphone devices

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

If devices appear, run:

```powershell
python app_continuous.py
```

Open:

```text
http://127.0.0.1:7860
```

---

## 4. ElevenLabs checks before demo

Before testing the full app, confirm your API and voice work.

### 4.1 Check API key

```bash
curl https://api.elevenlabs.io/v1/user \
  -H "xi-api-key: YOUR_API_KEY"
```

### 4.2 List voices

```bash
curl https://api.elevenlabs.io/v1/voices \
  -H "xi-api-key: YOUR_API_KEY"
```

Copy the cloned `voice_id`.

### 4.3 Test tiny TTS

```bash
curl -X POST "https://api.elevenlabs.io/v1/text-to-speech/YOUR_VOICE_ID/stream" \
  -H "xi-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a quick API test.",
    "model_id": "eleven_flash_v2_5"
  }' \
  --output test.mp3
```

If `test.mp3` plays, the API and voice are working.

---

## 5. Recommended model settings

For live/serial demo:

```text
Model: eleven_flash_v2_5
Stability: 0.45
Similarity Boost: 0.90
```

For slightly better quality but more delay:

```text
Model: eleven_turbo_v2_5
Stability: 0.45
Similarity Boost: 0.90
```

For polished non-realtime generation:

```text
Model: eleven_multilingual_v2
```

Recommended VAD settings:

```text
VAD Aggressiveness: 2
Speech End Silence: 900–1200 ms
Minimum Speech: 700–1000 ms
```

---

## 6. Add recorded voice clip generation tab

This feature lets you record a short clip or upload an audio file, then generate cloned voice from the same spoken content.

Workflow:

```text
record/upload audio clip → ElevenLabs STT → transcript → ElevenLabs TTS using cloned Voice ID → output WAV
```

Add this function to `app_continuous.py` if it is not already present:

```python
def recorded_clip_to_cloned_voice(
    audio_path,
    api_key_input,
    voice_id,
    model_id,
    stability,
    similarity_boost,
):
    api_key = get_api_key(api_key_input)
    voice_id = (voice_id or "").strip()

    if not audio_path:
        raise gr.Error("Please record or upload an audio clip first.")

    if not voice_id:
        raise gr.Error("Please paste Voice ID.")

    transcript = transcribe_audio_file(audio_path, api_key)

    if not transcript:
        raise gr.Error("No transcript detected from the clip.")

    output_path = tts_to_wav_file(
        text=transcript,
        voice_id=voice_id,
        api_key=api_key,
        model_id=model_id,
        stability=stability,
        similarity_boost=similarity_boost,
    )

    return output_path, transcript
```

Then add this tab inside the `with gr.Blocks(...) as demo:` section:

```python
with gr.Tab("Recorded Clip"):
    clip_input = gr.Audio(
        label="Record or upload voice clip",
        sources=["microphone", "upload"],
        type="filepath",
    )

    clip_btn = gr.Button("Generate Cloned Voice from Clip")

    clip_audio_output = gr.Audio(
        label="Cloned Voice Output",
        type="filepath",
        autoplay=True,
    )

    clip_transcript_output = gr.Textbox(
        label="Transcript",
        lines=4,
    )

    clip_btn.click(
        fn=recorded_clip_to_cloned_voice,
        inputs=[
            clip_input,
            api_key_input,
            voice_id_input,
            model_id_input,
            stability_input,
            similarity_input,
        ],
        outputs=[
            clip_audio_output,
            clip_transcript_output,
        ],
    )
```

Recommended clip length:

```text
3–10 seconds for quick testing
10–30 seconds for better demo samples
```

---

## 7. Demo workflow

### Text test

Use this first to verify voice quality:

```text
Type sentence → Generate → cloned voice plays
```

### Recorded clip test

Use this before continuous mode:

```text
Record short sentence → Generate → transcript appears → cloned voice plays
```

### Continuous mode

Use this for serial live demo:

```text
Start Continuous Mic
Speak one clear sentence
Pause briefly
Cloned voice plays
Speak next sentence
Pause briefly
Cloned voice plays
Stop
Download merged audio if needed
```

Tell the speaker:

```text
Please speak one sentence, then pause for about one second.
```

---

## 8. Common issues

### PortAudio library not found

Linux:

```bash
sudo apt install -y portaudio19-dev libportaudio2 libportaudiocpp0
pip uninstall -y sounddevice
pip install sounddevice
```

### No microphone detected

Check devices:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

On Windows, check:

```text
Settings → System → Sound → Input
```

Make sure the correct mic is selected and not muted.

### App opens but mic is not from the remote browser

This is expected for the local-microphone version.

The mic is always from the machine running Python.

### ElevenLabs API error 401

Check that the header is correct:

```text
xi-api-key
```

Check that `.env` is loaded and the key is not copied with extra spaces.

### Voice not found

Check `/v1/voices` and confirm the cloned voice ID belongs to the same ElevenLabs account/API key.

### Continuous mode cuts off too early

Increase:

```text
Speech End Silence: 1000–1200 ms
```

### Continuous mode captures noise

Increase:

```text
VAD Aggressiveness: 3
Minimum Speech: 1000 ms
```

### Output delay too long

Use:

```text
eleven_flash_v2_5
```

Keep sentences short.

---

## 9. Recommended final demo setup on Windows

```text
Windows laptop
├── local microphone input
├── Python app running locally
├── Gradio UI opened at http://127.0.0.1:7860
└── speakers/headphones for cloned voice output
```

For Teams/Zoom/OBS later, add a virtual audio router such as VB-CABLE or Voicemeeter.

---

## 10. Migration checklist

Before migration:

```text
[ ] Linux version runs
[ ] API key works
[ ] Voice ID works
[ ] Text generation works
[ ] Recorded clip generation works
[ ] Continuous mic detects speech
[ ] Cloned voice output plays
[ ] Merged audio works
```

After migration to Windows:

```text
[ ] Python installed
[ ] venv created
[ ] requirements installed
[ ] .env created
[ ] sounddevice sees Windows mic
[ ] Text generation works
[ ] Recorded clip generation works
[ ] Continuous mic works
```
