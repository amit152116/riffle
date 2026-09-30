import csv, subprocess, sys, numpy as np
SR = 8000; HOP = 80  # 10 ms envelope

def env(path, start, dur):
    raw = subprocess.run(["ffmpeg","-loglevel","error","-ss",str(start),"-t",str(dur),"-i",path,
                          "-ac","1","-ar",str(SR),"-f","s16le","-"],capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(float)
    n = len(x)//HOP
    e = np.sqrt((x[:n*HOP].reshape(n,HOP)**2).mean(1)+1)
    return np.log(e)

def refine(yt, ref, start, span=1.5, win=20.0):
    """Best start time in [start-span, start+span] aligning yt's envelope to ref's first `win` s."""
    r = env(ref, 0, win); r = r - r.mean()
    lo = max(0.0, start - span)
    y = env(yt, lo, win + 2*span)
    best = (-9, start)
    for k in range(0, int(2*span*100)+1):
        seg = y[k:k+len(r)]
        if len(seg) < len(r): break
        s = seg - seg.mean()
        c = float((s*r).sum()/ (np.linalg.norm(s)*np.linalg.norm(r)+1e-9))
        if c > best[0]: best = (c, lo + k/100)
    return best

if __name__ == "__main__":
    g = sys.argv[1]
    r = next(x for x in csv.DictReader(open("/home/amit_152116/riffle-trim-report.csv")) if x["group"]==g and x["flag"]=="trim")
    c, t = refine(r["yt_file"], r["reference"], float(r["keep_from"]))
    print(f"g{g}: fingerprint start {r['keep_from']}s -> refined {t:.2f}s (corr {c:.3f}, shift {t-float(r['keep_from']):+.2f}s)")
