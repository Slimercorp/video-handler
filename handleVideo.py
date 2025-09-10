import cv2, numpy as np, matplotlib.pyplot as plt
import pandas as pd

# ---------------- User settings ----------------
video_path = 'video1.mov'

# If the container doesn't report FPS correctly, fallback to this value:
fallback_fps = 100.0

# Process range in seconds (hardcoded). Use None for "till the end".
start_sec = 0
end_sec   = None  # e.g. 9.5
# ------------------------------------------------

cap = cv2.VideoCapture(video_path)
ok, first = cap.read()
if not ok:
    raise RuntimeError("Failed to open video")

# 1) ROI selection (thin strip across the motion)
WIN_ROI = "Select ROI - thin strip across motion"
rx, ry, rw, rh = map(int, cv2.selectROI(WIN_ROI, first, False, False))
cv2.destroyWindow(WIN_ROI)

# 2) Calibration: click two ruler marks with known distance (e.g. 10 mm)
WIN_CAL = "Calibration - click 2 ruler marks with distance 10 mm, then press any key"
pts = []
frame_for_clicks = first.copy()

def on_mouse(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        cv2.circle(frame_for_clicks, (x,y), 4, (0,0,255), -1)
        cv2.imshow(WIN_CAL, frame_for_clicks)

cv2.imshow(WIN_CAL, frame_for_clicks)
cv2.setMouseCallback(WIN_CAL, on_mouse)
print("Click two ruler marks (known distance, e.g. 10 mm), then press any key…")
cv2.waitKey(0); cv2.destroyAllWindows()
assert len(pts) >= 2, "Need two clicks for calibration."

px = np.hypot(pts[0][0]-pts[1][0], pts[0][1]-pts[1][1])
known_mm = 10.0
mm_per_px = known_mm / px

# FPS and frame range
fps_file = cap.get(cv2.CAP_PROP_FPS)
fps = fps_file if fps_file and fps_file > 1e-3 else fallback_fps

n_frames_prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)
n_frames = int(n_frames_prop) if n_frames_prop and n_frames_prop > 0 else None

start_f = int((start_sec or 0.0) * fps)
if end_sec is None:
    end_f = (n_frames - 1) if n_frames is not None else start_f + int(12 * fps)  # default span ~12s
else:
    end_f = int(end_sec * fps)

if n_frames is not None:
    start_f = max(0, min(start_f, n_frames - 1))
    end_f   = max(0, min(end_f,   n_frames - 1))
if start_f > end_f:
    start_f, end_f = end_f, start_f

# Seek once to start, then read sequentially
cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

positions_px = []
frames_to_read = end_f - start_f + 1
for i in range(frames_to_read):
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = gray[ry:ry+rh, rx:rx+rw]

    # Vertical motion case (ROI is a vertical thin strip): average across width -> profile along Y
    profile = crop.mean(axis=1)  # shape = (rh,)

    g = np.gradient(profile.astype(np.float64))
    idx = int(np.clip(np.argmax(np.abs(g)), 1, len(profile)-2))

    # Subpixel: quadratic fit around the peak (idx-1, idx, idx+1)
    xs = np.array([idx-1, idx, idx+1], dtype=float)
    ys = profile[[idx-1, idx, idx+1]].astype(float)
    a, b, c = np.polyfit(xs, ys, 2)
    sub = -b/(2*a) if a != 0 else float(idx)
    positions_px.append(sub)

cap.release()

positions_px = np.array(positions_px)
t = np.arange(len(positions_px)) / fps
y_mm = (positions_px - positions_px[0]) * mm_per_px

# Light smoothing (moving average)
win = 7
kernel = np.ones(win)/win
y_mm_smooth = np.convolve(y_mm, kernel, mode='same')

# Rough frequency estimate via FFT
freq = np.fft.rfftfreq(len(t), d=1/fps)
amp = np.abs(np.fft.rfft(y_mm_smooth - y_mm_smooth.mean()))
f0 = freq[amp.argmax()] if len(freq) else np.nan
print(f"Estimated frequency: ~{f0:.3f} Hz")
print(f"mm_per_px = {mm_per_px:.6f} mm/px (known_mm={known_mm} mm)")
print(f"Frames processed: {len(positions_px)}  |  segment: start={start_f}, end={end_f}  |  FPS used: {fps:.3f}")

# Save CSV and plot
pd.DataFrame({"t_s": t, "y_mm": y_mm, "y_mm_smooth": y_mm_smooth}).to_csv("beam_y_vs_t.csv", index=False)

plt.figure()
plt.plot(t, y_mm, lw=1, label='raw')
plt.plot(t, y_mm_smooth, lw=1.5, label='smoothed')
plt.xlabel('t, s'); plt.ylabel('y, mm'); plt.legend(); plt.tight_layout(); plt.show()
