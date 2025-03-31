import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================
# CONFIGURATION
# ========================
VIDEO_PATH = r"C:\Users\aravi\bit_rate vs dip_switch.mkv"
NOISE_LEVELS = [0, 5, 10, 15, 20]  # Noise levels in percentage
FRAME_SKIP = 10                    # Process every 10th frame for speed
EXPECTED_BITS = 10000              # Expected number of bits per frame (adjust for your setup)

# ========================
# FUNCTIONS
# ========================
def inject_noise(frame, noise_level):
    """Inject noise into a frame based on the given noise level."""
    noise = np.random.randint(0, 256, frame.shape, dtype=np.uint8)
    noisy_frame = cv2.addWeighted(frame, 1 - noise_level / 100, noise, noise_level / 100, 0)
    return noisy_frame

def calculate_ber(original_frame, noisy_frame):
    """Calculate Bit Error Rate (BER) between original and noisy frames."""
    original_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    noisy_gray = cv2.cvtColor(noisy_frame, cv2.COLOR_BGR2GRAY)
    
    _, original_binary = cv2.threshold(original_gray, 127, 1, cv2.THRESH_BINARY_INV)
    _, noisy_binary = cv2.threshold(noisy_gray, 127, 1, cv2.THRESH_BINARY_INV)
    
    errors = np.sum(original_binary != noisy_binary)
    total_bits = original_binary.size
    return errors / total_bits

# ========================
# MAIN PROCESSING
# ========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video file: {VIDEO_PATH}")

ber_results = {level: [] for level in NOISE_LEVELS}
frame_count = 0

print("Processing video for BER analysis...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Skip frames for speed
    if frame_count % FRAME_SKIP != 0:
        frame_count += 1
        continue
    
    # Inject noise and calculate BER for each noise level
    for level in NOISE_LEVELS:
        noisy_frame = inject_noise(frame, level)
        ber = calculate_ber(frame, noisy_frame)
        ber_results[level].append(ber)
    
    frame_count += 1

cap.release()

# ========================
# AGGREGATE RESULTS & PLOT
# ========================
average_ber = {level: np.mean(ber_results[level]) for level in NOISE_LEVELS}

plt.figure(figsize=(10, 6))
plt.plot(NOISE_LEVELS, list(average_ber.values()), marker='o', color='blue')
plt.title('Bit Error Rate (BER) vs. Noise Levels')
plt.xlabel('Noise Level (%)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True)
plt.xticks(NOISE_LEVELS)
plt.tight_layout()
plt.savefig('ber_vs_noise.png', dpi=300)
plt.show()

print("BER analysis complete. Results:")
for level in NOISE_LEVELS:
    print(f"Noise Level {level}%: Average BER = {average_ber[level]:.4f}")
