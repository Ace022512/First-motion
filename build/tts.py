import sys, json, soundfile as sf, numpy as np
from kokoro_onnx import Kokoro
K = sys.argv[1]; out = sys.argv[2]; voice = sys.argv[3] if len(sys.argv)>3 else "am_michael"
lines = json.load(open(sys.argv[4] if len(sys.argv)>4 else "build/script.json"))
k = Kokoro(f"{K}/kokoro-v1.0.onnx", f"{K}/voices-v1.0.bin")
meta=[]
for i,l in enumerate(lines):
    s, sr = k.create(l["say"], voice=voice, speed=l.get("speed",1.08), lang="en-us")
    # trim silence
    idx = np.where(np.abs(s)>0.01)[0]; s = s[max(0,idx[0]-800):idx[-1]+1600]
    sf.write(f"{out}/l{i:02d}.wav", s, sr); meta.append(len(s)/sr)
    print(i, round(len(s)/sr,2), l["say"])
print("total", round(sum(meta),2))
json.dump(meta, open(f"{out}/durs.json","w"))
