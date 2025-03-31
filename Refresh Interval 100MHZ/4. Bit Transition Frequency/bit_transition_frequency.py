import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuration
VIDEO_PATH = r"C:\Users\aravi\100mhz dipswtich.mkv"
EXPECTED_FREQ = 2  # Expected blinking rate in Hz (from your FPGA code)
FRAME_SKIP = 3     # Process every 3rd frame (adjust for speed/accuracy tradeoff)

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Pre-allocate arrays
transition_series = []
prev_frame = None
frame_count = 0

print("Processing video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Skip frames for faster processing
    if frame_count % FRAME_SKIP != 0:
        frame_count += 1
        continue
    
    # Convert to binary (black square = 1, white = 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
    
    # Calculate transitions using NumPy XOR for speed
    if prev_frame is not None:
        transitions = np.logical_xor(binary, prev_frame).sum()
        transition_series.append(transitions)
    
    prev_frame = binary.copy()
    frame_count += 1

cap.release()
print(f"Processed {len(transition_series)} frames")

# FFT computation
N = len(transition_series)
window = np.hanning(N)
sampling_rate = fps / FRAME_SKIP  # Adjusted for frame skipping

fft = np.fft.rfft(transition_series * window)
freqs = np.fft.rfftfreq(N, d=1/sampling_rate)
magnitude = np.abs(fft)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(freqs, magnitude, color='blue', label='FFT Magnitude')
plt.axvline(EXPECTED_FREQ, color='red', linestyle='--', 
           label=f'Expected: {EXPECTED_FREQ} Hz')
plt.title(f"Bit Transition Frequency Analysis\n(Video: {total_frames/fps:.1f}s @ {fps:.0f}FPS)")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.xlim(0, sampling_rate/2)  # Nyquist limit

# Annotate peaks
peaks = np.argsort(magnitude)[-3:]  # Top 3 frequencies
for p in peaks:
    if freqs[p] > 0.1:  # Ignore DC component
        plt.annotate(f'{freqs[p]:.1f} Hz\n({magnitude[p]:.0f})',
                     (freqs[p], magnitude[p]),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center',
                     color='darkgreen')

plt.tight_layout()
plt.savefig('fft_analysis.png', dpi=300)
plt.show()
