"""Render the Ballers Bridge 30s motion video (1080x1920, 30fps) with voiceover + burned-in captions.

Usage: python3 build/render.py            -> out/ballers_bridge_30s.mp4 (+ .srt)
       python3 build/render.py --still T   -> writes a single frame at time T for preview
"""
import json, math, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS, TOTAL = 1080, 1920, 30, 30.0
YEL = (255, 196, 22)
WHITE = (255, 255, 255)
FONT_DIR = "/usr/share/fonts/opentype/inter/"


def font(weight, size):
    return ImageFont.truetype(FONT_DIR + f"Inter-{weight}.otf", size)


# ---------------------------------------------------------------- assets
poster = Image.open(f"{ROOT}/assets/poster.png").convert("RGB")


def keyed(box, lo=45, hi=110):
    """Crop from poster and turn the dark background transparent."""
    im = poster.crop(box)
    a = np.asarray(im).astype(float).max(2)
    alpha = np.clip((a - lo) / (hi - lo), 0, 1)
    h, w = alpha.shape  # feather crop edges so no hard rectangle shows
    fy = np.clip(np.minimum(np.arange(h), h - 1 - np.arange(h)) / 6, 0, 1)
    fx = np.clip(np.minimum(np.arange(w), w - 1 - np.arange(w)) / 6, 0, 1)
    alpha = alpha * fy[:, None] * fx[None, :] * 255
    out = im.convert("RGBA")
    out.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    return out


def up(im, scale):
    return im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)


LOGO = keyed((41, 18, 285, 137))
_la = LOGO.getchannel("A"); ImageDraw.Draw(_la).rectangle((0, 0, 45, 45), fill=0); LOGO.putalpha(_la)  # drop gym light
DROP = keyed((511, 675, 922, 763), lo=35, hi=80)
BUILT = keyed((774, 836, 924, 919), lo=70, hi=140)
ICONS = [keyed(b, lo=60, hi=140) for b in
         [(607, 322, 670, 374), (607, 410, 670, 467), (607, 503, 675, 555), (607, 592, 670, 649)]]

# background: the player / title side of the poster
PLAYER = poster.crop((0, 0, 560, 960))
# remove the baked-in logo (top-left) and footer strip so animated versions don't double up
_m = Image.new("L", PLAYER.size, 255)
_d = ImageDraw.Draw(_m)
_d.rectangle((0, 0, 360, 185), fill=0)
_d.rectangle((0, 845, 560, 960), fill=0)
PLAYER = Image.composite(PLAYER, Image.new("RGB", PLAYER.size, (6, 6, 6)), _m.filter(ImageFilter.GaussianBlur(28)))
PLAYER_BIG = PLAYER.resize((1120, 1920), Image.LANCZOS)
PLAYER_DIM = Image.blend(PLAYER_BIG.filter(ImageFilter.GaussianBlur(10)),
                         Image.new("RGB", PLAYER_BIG.size, (0, 0, 0)), 0.62)

# ---------------------------------------------------------------- script / timing
script = json.load(open(f"{ROOT}/build/script.json"))
durs = json.load(open(f"{ROOT}/build/audio/durs.json"))
# pause after each line (seconds); bigger pauses at section changes
gaps = [0.4, 0.2, 0.4, 0.45, 0.2, 0.2, 0.2, 0.4, 0.2, 0.4, 0.0]
LEAD = 0.35
starts, t = [], LEAD
for d, g in zip(durs, gaps):
    starts.append(t)
    t += d + g
ends = [s + d for s, d in zip(starts, durs)]
assert t < TOTAL - 1.2, t


def word_times(i):
    """Approximate per-word timings within a spoken line, weighted by length + punctuation."""
    words = script[i]["cap"].split()
    wts = [len(w.strip("—")) + 1.5 + (3 if w[-1] in ",." else 0) + (4 if w in ("A", "B", "C", "D", "—") else 0)
           for w in words]
    tot, acc, out = sum(wts), 0, []
    for w, k in zip(words, wts):
        out.append((w, starts[i] + durs[i] * acc / tot))
        acc += k
    return out


def chunks(i, maxc=34):
    """Split a caption line into on-screen chunks: (start, end, [(word, t)])."""
    wt = word_times(i)
    groups, cur = [], []
    for w, ts in wt:
        cur.append((w, ts))
        txt = " ".join(x for x, _ in cur)
        if (w[-1] in ",." and len(txt) > 14) or len(txt) > maxc:
            groups.append(cur); cur = []
    if cur:
        if len(cur) <= 2 and groups:  # don't orphan a trailing word
            groups[-1] += cur
        else:
            groups.append(cur)
    out = []
    for gi, g in enumerate(groups):
        s = g[0][1]
        e = groups[gi + 1][0][1] if gi + 1 < len(groups) else ends[i] + min(gaps[i], 0.35)
        out.append((s, e, g))
    return out


CAPS = [c for i in range(len(script)) for c in chunks(i)]


# ---------------------------------------------------------------- easing helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x); return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x); return x * x * (3 - 2 * x)


def back_out(x, s=1.9):
    x = clamp(x); x -= 1; return x * x * ((s + 1) * x + s) + 1


def prog(t, t0, dur=0.45):
    return clamp((t - t0) / dur)


# ---------------------------------------------------------------- drawing helpers
def paste(canvas, im, cx, cy, alpha=1.0, scale=1.0):
    if alpha <= 0.003:
        return
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if alpha < 1:
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im = im.copy(); im.putalpha(a)
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


def text_img(txt, f, fill=WHITE, stroke=0, shadow=True):
    l, t, r, b = f.getbbox(txt, stroke_width=stroke)
    pad = 24
    im = Image.new("RGBA", (r - l + pad * 2, b - t + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad - l, pad - t + 4), txt, font=f, fill=(0, 0, 0, 200), stroke_width=stroke + 2)
        im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6)))
    d.text((pad - l, pad - t), txt, font=f, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0))
    return im


_tcache = {}


def T(txt, weight, size, fill=WHITE):
    k = (txt, weight, size, fill)
    if k not in _tcache:
        _tcache[k] = text_img(txt, font(weight, size), fill)
    return _tcache[k]


def rise(canvas, im, cx, cy, t, t0, t_out=None, dy=50):
    """Slide-up + fade in at t0; fade out at t_out."""
    p = ease_out(prog(t, t0, 0.5))
    a = p
    if t_out is not None:
        a *= 1 - ease_io(prog(t, t_out, 0.35))
    paste(canvas, im, cx, cy + (1 - p) * dy, a)


# ---------------------------------------------------------------- option cards
CARD_TXT = [("Getting college", "coaches to notice you"), ("Finding schools", "that fit your goals"),
            ("Knowing what to", "send coaches"), ("Understanding the", "recruiting process")]
_card_cache = {}


def card(i, hot):
    k = (i, hot)
    if k in _card_cache:
        return _card_cache[k]
    cw, ch, r = 900, 170, 26
    im = Image.new("RGBA", (cw + 60, ch + 60), (0, 0, 0, 0))
    if hot:  # glow
        g = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).rounded_rectangle((30, 30, 30 + cw, 30 + ch), r, outline=YEL + (230,), width=14)
        im.alpha_composite(g.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((30, 30, 30 + cw, 30 + ch), r,
                        fill=(40, 30, 6, 235) if hot else (8, 8, 8, 215), outline=YEL, width=5 if hot else 3)
    # letter badge
    cx, cy = 30 + 95, 30 + ch // 2
    d.ellipse((cx - 50, cy - 50, cx + 50, cy + 50), fill=YEL)
    lf = font("Black", 58)
    d.text((cx, cy + 2), "ABCD"[i], font=lf, fill=(10, 10, 10), anchor="mm")
    # icon
    ic = up(ICONS[i], 1.5)
    im.alpha_composite(ic, (30 + 190, cy - ic.height // 2))
    # text
    tf = font("SemiBold", 46)
    d.text((30 + 315, cy - 30), CARD_TXT[i][0], font=tf, fill=WHITE, anchor="lm")
    d.text((30 + 315, cy + 30), CARD_TXT[i][1], font=tf, fill=WHITE, anchor="lm")
    _card_cache[k] = im
    return im


# ---------------------------------------------------------------- captions
CAP_F = font("ExtraBold", 54)
CAP_Y = 1470


def draw_caption(canvas, t):
    cur = [c for c in CAPS if c[0] <= t < c[1]]
    if not cur:
        return
    s, e, words = cur[-1]
    a = clamp((t - s) / 0.12) * clamp((e - t) / 0.12)
    # layout words into max 2 rows of <= 26 chars
    rows, row = [], []
    for w, ts in words:
        if row and len(" ".join(x for x, _ in row + [(w, ts)])) > 24:
            rows.append(row); row = []
        row.append((w, ts))
    rows.append(row)
    sp = CAP_F.getlength(" ")
    lh = 72
    total_h = lh * len(rows)
    layer = Image.new("RGBA", (W, total_h + 80), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    widths = [sum(CAP_F.getlength(w) for w, _ in r) + sp * (len(r) - 1) for r in rows]
    bw = max(widths) + 70
    d.rounded_rectangle(((W - bw) / 2, 10, (W + bw) / 2, total_h + 62), 22, fill=(0, 0, 0, 165))
    y = 36
    for r, rw in zip(rows, widths):
        x = (W - rw) / 2
        for w, ts in r:
            active = ts <= t
            nxt = [x2 for x2 in words if x2[1] > ts]
            current = active and (not nxt or nxt[0][1] > t)
            col = YEL if current else (WHITE if active else (200, 200, 200))
            d.text((x, y), w, font=CAP_F, fill=col + (255,), stroke_width=3, stroke_fill=(0, 0, 0))
            x += CAP_F.getlength(w) + sp
        y += lh
    if a < 1:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * a)))
    canvas.alpha_composite(layer, (0, CAP_Y - 10))


# ---------------------------------------------------------------- scenes
S = starts
SCENE_B = S[1] - 0.25        # intro text
SCENE_C_OUT = S[4] - 0.3     # end of question
SCENE_D_OUT = S[8] - 0.25    # end of options
SCENE_E_OUT = S[10] - 0.3    # end of "drop your answer"


def background(t):
    # Opening: sharp poster photo w/ title, slow push-in; then crossfade to dimmed blurred version.
    z = 1.0 + 0.06 * (t / TOTAL)
    mix = ease_io((t - (SCENE_B - 0.3)) / 0.7)
    if t < 1.6:  # opening push from close-up of title
        z = 1.0 + 0.22 * (1 - ease_out(t / 1.6))
    if mix <= 0:
        src = PLAYER_BIG
    elif mix >= 1:
        src = PLAYER_DIM
    else:
        src = Image.blend(PLAYER_BIG, PLAYER_DIM, mix)
    cw, ch = W / z, H / z
    # focus slightly toward title in the opening
    fx = 560 if t < 2 else 560
    fy = 1150 if t < 2.5 else 960
    x0 = clamp(fx - cw / 2, 0, src.width - cw)
    y0 = clamp(fy - ch / 2, 0, src.height - ch)
    bg = src.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
    return bg.convert("RGBA")


VIG = None


def vignette():
    global VIG
    if VIG is None:
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2)
        a = np.clip((r - 0.45) * 1.4, 0, 0.8) * 255
        # extra darkening behind caption zone
        a = np.maximum(a, np.clip((yy - 1300) / 400, 0, 1) * 150)
        v = np.zeros((H, W, 4), np.uint8); v[..., 3] = a.astype(np.uint8)
        VIG = Image.fromarray(v, "RGBA")
    return VIG


def frame(t):
    c = background(t)
    c.alpha_composite(vignette())

    # ---- Scene A: "Let's talk recruiting" (title lives in the photo). Logo pops top.
    if t < SCENE_B + 0.5:
        pass
    # persistent logo (top) from opening until outro
    if t < TOTAL - 3.0:
        p = back_out(prog(t, 0.15, 0.6))
        a = clamp(p) * (1 - ease_io(prog(t, S[10] - 0.5, 0.4)))
        paste(c, up(LOGO, 1.55), W / 2, 150, a, max(0.01, 0.6 + 0.4 * p))

    # ---- Scene B: You've put in the work...
    out_b = S[3] - 0.35
    if SCENE_B <= t < out_b + 0.5:
        rise(c, T("You’ve put in the work.", "Black", 76), W / 2, 690, t, S[1], out_b)
        wt = word_times(2)
        lines = [("You’re getting shots up,", 0), ("hitting the weight room,", 4), ("and chasing the dream.", 8)]
        for k, (txt, wi) in enumerate(lines):
            col = YEL if k == 2 else WHITE
            rise(c, T(txt, "SemiBold", 60, col), W / 2, 850 + k * 92, t, wt[wi][1] - 0.1, out_b)

    # ---- Scene C: question
    if S[3] - 0.3 <= t < SCENE_C_OUT + 0.5:
        wt = word_times(3)
        rise(c, T("But when it comes to", "Bold", 64), W / 2, 640, t, wt[0][1] - 0.1, SCENE_C_OUT)
        rise(c, T("college recruiting,", "Black", 84, YEL), W / 2, 750, t, wt[5][1] - 0.15, SCENE_C_OUT)
        p = back_out(prog(t, wt[7][1] - 0.1, 0.55))
        if p > 0:
            a = clamp(p) * (1 - ease_io(prog(t, SCENE_C_OUT, 0.35)))
            paste(c, T("what’s the biggest", "Black", 84), W / 2, 910, a, 0.7 + 0.3 * p)
            paste(c, T("headache?", "Black", 132, YEL), W / 2, 1040, a, 0.6 + 0.4 * p)

    # ---- Scene D: options
    if S[4] - 0.4 <= t < SCENE_D_OUT + 0.5:
        a_hdr = ease_out(prog(t, S[4] - 0.4, 0.4)) * (1 - ease_io(prog(t, SCENE_D_OUT, 0.35)))
        paste(c, T("WHAT’S YOUR BIGGEST HEADACHE?", "ExtraBold", 44, YEL), W / 2, 380, a_hdr)
        for i in range(4):
            ti = S[4 + i]
            p = ease_out(prog(t, ti - 0.15, 0.45))
            if p <= 0:
                continue
            hot = ti - 0.15 <= t < (S[5 + i] - 0.15 if i < 3 else SCENE_D_OUT)
            a = p * (1 - ease_io(prog(t, SCENE_D_OUT + i * 0.05, 0.35)))
            sc = 1.0
            if hot:
                sc = 1.0 + 0.035 * ease_out(prog(t, ti, 0.3))
            x = W / 2 + (1 - p) * 600
            paste(c, card(i, hot), x, 540 + i * 215, a if hot else a * 0.85, sc)

    # ---- Scene E: drop your answer
    if S[8] - 0.3 <= t < SCENE_E_OUT + 0.5:
        p = back_out(prog(t, S[8] - 0.2, 0.55), 2.4)
        a = clamp(p * 1.5) * (1 - ease_io(prog(t, SCENE_E_OUT, 0.35)))
        wob = math.sin((t - S[8]) * 2.2) * 0.012
        paste(c, up(DROP, 2.45), W / 2, 640, a, max(0.01, 0.5 + 0.5 * p + wob))
        wt = word_times(9)
        rise(c, T("We want to hear the real story,", "Bold", 58), W / 2, 870, t, wt[0][1] - 0.1, SCENE_E_OUT)
        rise(c, T("not the highlight-reel version.", "Bold", 58, YEL), W / 2, 950, t, wt[6][1] - 0.1, SCENE_E_OUT)

    # ---- Scene F: outro
    if t >= S[10] - 0.35:
        k = ease_io(prog(t, S[10] - 0.35, 0.6))
        c.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(140 * k))))
        p = back_out(prog(t, S[10] - 0.3, 0.7))
        paste(c, up(LOGO, 3.4), W / 2, 600, clamp(p), max(0.01, 0.75 + 0.25 * p))
        p2 = back_out(prog(t, S[10] + 0.9, 0.6), 2.2)
        if p2 > 0:
            paste(c, up(BUILT, 3.2), W / 2, 960, clamp(p2), max(0.01, 0.4 + 0.6 * p2))
        p3 = ease_out(prog(t, S[10] + 1.6, 0.6))
        if p3 > 0:
            tags = T("ATHLETES  |  FAMILIES  |  COACHES", "Bold", 40)
            paste(c, tags, W / 2, 1200, p3)
            paste(c, T("A STRONGER RECRUITING JOURNEY", "SemiBold", 34, YEL), W / 2, 1265, p3 * 0.95)

    draw_caption(c, t)

    # global fade in/out
    fade = min(clamp(t / 0.35), clamp((TOTAL - t) / 0.6))
    if fade < 1:
        c = Image.blend(Image.new("RGBA", c.size, (0, 0, 0, 255)), c, fade)
    return c.convert("RGB")


# ---------------------------------------------------------------- audio / srt / encode
def build_audio(path):
    sr = 24000
    track = np.zeros(int(TOTAL * sr), np.float32)
    import soundfile as sf
    for i, s in enumerate(starts):
        a, r = sf.read(f"{ROOT}/build/audio/l{i:02d}.wav", dtype="float32")
        assert r == sr
        k = int(s * sr)
        track[k:k + len(a)] += a[: len(track) - k]
    track /= max(1e-6, np.abs(track).max()) / 0.89
    sf.write(path, track, sr)


def srt_time(x):
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def build_srt(path):
    with open(path, "w") as f:
        for n, (s, e, words) in enumerate(CAPS, 1):
            f.write(f"{n}\n{srt_time(s)} --> {srt_time(e)}\n{' '.join(w for w, _ in words)}\n\n")


def main():
    os.makedirs(f"{ROOT}/out", exist_ok=True)
    if "--still" in sys.argv:
        for ts in sys.argv[sys.argv.index("--still") + 1:]:
            frame(float(ts)).save(f"{ROOT}/out/still_{float(ts):05.2f}.png")
        return
    wav = f"{ROOT}/build/audio/voice_mix.wav"
    build_audio(wav)
    build_srt(f"{ROOT}/out/ballers_bridge_30s.srt")
    out = f"{ROOT}/out/ballers_bridge_30s.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(TOTAL * FPS)
    for i in range(n):
        p.stdin.write(frame(i / FPS).tobytes())
        if i % 90 == 0:
            print(f"frame {i}/{n}", flush=True)
    p.stdin.close(); p.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
