"""Procedurally synthesized basketball-gym sound bed: room tone, ball dribbles,
sneaker squeaks, a rim/net swish, all through a big-gym reverb, ducked under the voice.
Pure numpy (FFT filtering / convolution), deterministic via a fixed seed."""
import numpy as np

SR = 48000


def fft_filter(x, lo=None, hi=None, sr=SR):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    g = np.ones_like(f)
    if lo:
        g *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** 4)
    if hi:
        g *= 1 / np.sqrt(1 + (f / hi) ** 4)
    return np.fft.irfft(X * g, len(x))


def convolve(x, ir):
    n = len(x) + len(ir) - 1
    N = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[:len(x)]


def gym_ir(rng, rt60=1.7, sr=SR):
    """Big hardwood gym: sparse early reflections + bright-ish diffuse tail."""
    n = int(rt60 * 1.2 * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail = fft_filter(tail, lo=150, hi=6000)
    tail[: int(0.02 * sr)] *= np.linspace(0, 1, int(0.02 * sr))
    ir = tail * 0.35
    for d, g in [(0.013, 0.5), (0.027, 0.38), (0.041, 0.3), (0.063, 0.22), (0.089, 0.16)]:
        ir[int(d * sr)] += g * rng.choice([-1, 1])
    ir[0] = 1.0
    return ir


def dribble(rng, sr=SR):
    """Ball hitting hardwood: low pitched thump + slap transient."""
    n = int(0.22 * sr)
    t = np.arange(n) / sr
    f0 = rng.uniform(95, 125)
    freq = f0 * (1 + 0.8 * np.exp(-t / 0.008))
    body = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t / 0.045)
    slap = fft_filter(rng.standard_normal(n), lo=250, hi=2500) * np.exp(-t / 0.006)
    ring = np.sin(2 * np.pi * rng.uniform(380, 460) * t) * np.exp(-t / 0.03) * 0.15
    return body * 1.0 + slap * 0.9 + ring


def squeak(rng, sr=SR):
    """Sneaker squeak: wobbling chirp with harmonics and a rough onset."""
    dur = rng.uniform(0.12, 0.26)
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = rng.uniform(1700, 2300) + rng.uniform(300, 700) * (t / dur) * rng.choice([-1, 1])
    f = f * (1 + 0.03 * np.sin(2 * np.pi * rng.uniform(25, 45) * t))
    ph = 2 * np.pi * np.cumsum(f) / sr
    s = np.sin(ph) + 0.45 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
    env = np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.03) * (0.7 + 0.3 * np.sin(np.pi * t / dur))
    grit = fft_filter(rng.standard_normal(n), lo=1500, hi=5000) * 0.25
    return (s + grit) * env * 0.5


def swish(rng, sr=SR):
    """Net swish: band-limited noise with a quick swell."""
    n = int(0.45 * sr)
    t = np.arange(n) / sr
    env = (t / 0.06) * np.exp(1 - t / 0.06)
    return fft_filter(rng.standard_normal(n), lo=1800, hi=9000) * env * 0.6


def place(track, clip, at, gain, sr=SR):
    k = int(at * sr)
    if k >= len(track):
        return
    m = min(len(clip), len(track) - k)
    track[k:k + m] += clip[:m] * gain


def gym_bed(total, voice, starts, seed=7, sr=SR):
    """voice: mono float array at `sr` (used to duck the bed). Returns the bed (same length)."""
    rng = np.random.default_rng(seed)
    n = int(total * sr)
    dry_far = np.zeros(n)   # distant action -> mostly reverb
    dry_near = np.zeros(n)  # closer dribbler -> more direct sound

    # Steady dribble (a player working on handles): ~2 bounces/sec with crossovers.
    t = 0.05
    while t < total - 0.6:
        g = rng.uniform(0.75, 1.0)
        place(dry_near, dribble(rng), t, g)
        t += rng.choice([0.46, 0.5, 0.54, 0.25], p=[0.3, 0.35, 0.25, 0.1])
    # Second ball, far court
    t = 0.3
    while t < total - 0.6:
        place(dry_far, dribble(rng), t, rng.uniform(0.5, 0.8))
        t += rng.uniform(0.55, 0.75)
    # Sneaker squeaks, scattered (clusters of 1-3)
    t = 0.9
    while t < total - 1.0:
        for _ in range(rng.integers(1, 4)):
            place(dry_far, squeak(rng), t, rng.uniform(0.35, 0.7))
            t += rng.uniform(0.15, 0.35)
        t += rng.uniform(1.4, 3.2)
    # Accents synced to the edit: swish on the outro, squeak hits on scene changes
    place(dry_near, swish(rng), starts[10] - 0.25, 0.9)
    for s in (starts[1], starts[3], starts[4], starts[8]):
        place(dry_far, squeak(rng), s - 0.3, 0.8)

    ir = gym_ir(rng)
    wet = convolve(dry_near * 0.5 + dry_far, ir)
    bed = dry_near * 0.6 + dry_far * 0.12 + wet * 0.28
    # room tone / HVAC hum
    room = fft_filter(rng.standard_normal(n), lo=60, hi=900) * 0.02
    bed += room
    bed /= np.abs(bed).max()

    return bed * duck_curve(voice, total, 0.6)


def duck_curve(voice, total, depth, sr=SR):
    """Gain curve that dips by `depth` while the voice is speaking, plus picture fade in/out."""
    n = int(total * sr)
    env = np.abs(voice[:n])
    win = int(0.12 * sr)
    env = np.convolve(env, np.ones(win) / win, mode="same")
    env = np.clip(env / (env.max() * 0.25), 0, 1)
    rel = int(0.35 * sr)  # slow release
    env = np.clip(np.convolve(env, np.ones(rel) / rel, mode="same") * 2, 0, 1)
    tt = np.arange(n) / sr
    fade = np.clip(tt / 0.4, 0, 1) * np.clip((total - tt) / 1.0, 0, 1)
    return (1 - depth * env) * fade
