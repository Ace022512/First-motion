# First-motion
Ballers Bridge 

## Ballers Bridge – "Let's Talk Recruiting" (30s motion video)

- `out/ballers_bridge_30s.mp4`: 1080×1920, 30 fps, 30 s, with voiceover, gym sound bed and burned-in captions
- `out/ballers_bridge_30s.srt`: caption file to upload separately

### Rebuild
```bash
pip install kokoro-onnx soundfile numpy pillow
# voice clips (needs kokoro-v1.0.onnx + voices-v1.0.bin from github.com/thewh1teagle/kokoro-onnx releases)
python3 build/tts.py <kokoro_model_dir> build/audio am_michael
python3 build/render.py               # full render (video + audio + srt)
python3 build/render.py --audio-only  # re-mix the audio only, keeping the video
python3 build/render.py --still 3.0 12.5   # preview frames
```
Edit the narration and caption text in `build/script.json`, and the gym sound level with `GYM_LEVEL` in `build/render.py`.
