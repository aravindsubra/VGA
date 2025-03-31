import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================
# CONFIGURATION
# ========================
VIDEO_PATH = r"C:\Users\aravi\bit_rate vs dip_switch.mkv"  # Your video path
FRAME_SKIP = 5                  # Process every 5th frame (1=all frames)
PLOT_FILE = "entropy_analysis.png"  # Output plot filename
ANOMALY_THRESHOLD = 2.0         # Sigma threshold for glitch detection

# ========================
# FUNCTIONS
# ========================
def frame_entropy(frame):
    """Calculate normalized Shannon entropy for a video frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / hist.sum() + 1e-12  # Normalize and prevent log(0)
    return -np.sum(hist * np.log2(hist)) / 8  # Normalized to 0-1

# ========================
# MAIN PROCESSING
# ========================
# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video file: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
times, entropies = [], []

print(f"Processing {VIDEO_PATH}...")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process every nth frame
    if frame_count % FRAME_SKIP == 0:
        entropy = frame_entropy(frame)
        entropies.append(entropy)
        times.append(frame_count / fps)
        print(f"Frame {frame_count}: Entropy = {entropy:.3f}")
    
    frame_count += 1

cap.release()
print(f"Processed {len(entropies)} frames")

# ========================
# ANALYSIS & PLOTTING
# ========================
# Calculate statistics
mean = np.mean(entropies)
std = np.std(entropies)
anomalies = np.where(np.abs(entropies - mean) > ANOMALY_THRESHOLD*std)[0]

# Create plot
plt.figure(figsize=(14, 6))
plt.plot(times, entropies, 'b-', lw=1, label='Frame Entropy')
plt.axhline(mean, color='r', linestyle='--', label=f'Mean ({mean:.2f} ± {std:.2f})')

# Annotate anomalies
for idx in anomalies:
    plt.annotate('GLITCH?', (times[idx], entropies[idx]),
                xytext=(0, 10), textcoords='offset points',
                ha='center', color='red', fontweight='bold')

# Format plot
plt.title(f"Video Entropy Analysis\n{VIDEO_PATH}", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Normalized Entropy (0-1)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, max(times))

# Save and show
plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
print(f"Plot saved to {PLOT_FILE}")
plt.show()

# ========================
# COMPRESSION ESTIMATE
# ========================
compression_potential = (1 - mean) * 100
print(f"\nCompression Potential Estimate: {compression_potential:.1f}%")
print(f"Anomalous Frames Detected: {len(anomalies)}")
