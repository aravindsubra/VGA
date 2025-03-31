import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================
# CONFIGURATION
# ========================
VIDEO_PATH = r"C:\Users\aravi\bit_rate vs dip_switch.mkv"
FRAME_SKIP = 5          # Process every 5th frame for speed
VOLTAGE = 1.2           # FPGA voltage reference (for power estimation)
OUTPUT_PLOT = "bitstream_autocorrelation.png"

# ========================
# VIDEO PROCESSING
# ========================
def get_bit_transitions():
    """Extract bit transition counts from video"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    transition_counts = []
    prev_binary = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for faster processing
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % FRAME_SKIP != 0:
            continue

        # Convert to binary (black square = 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
        
        # Calculate transitions from previous frame
        if prev_binary is not None:
            transitions = np.sum(binary != prev_binary)
            transition_counts.append(transitions)
            
        prev_binary = binary.copy()

    cap.release()
    return np.array(transition_counts)

# ========================
# AUTOCORRELATION ANALYSIS
# ========================
def compute_autocorrelation(signal):
    """FFT-based autocorrelation with normalization"""
    n = len(signal)
    mean = np.mean(signal)
    signal_centered = signal - mean
    corr = np.correlate(signal_centered, signal_centered, mode='full')
    return corr[corr.size//2:] / (np.var(signal) * np.arange(n, 0, -1))

# ========================
# PLOTTING & EXECUTION
# ========================
if __name__ == "__main__":
    print("Processing video...")
    transitions = get_bit_transitions()
    
    print("Computing autocorrelation...")
    autocorr = compute_autocorrelation(transitions)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(autocorr, color='darkorange', linewidth=1)
    plt.title(f"Bitstream Autocorrelation\n(Detected Periodicity: {np.argmax(autocorr[1:])+1} frames)")
    plt.xlabel("Time Lag (Frames)")
    plt.ylabel("Normalized Correlation")
    plt.grid(True, alpha=0.3)
    
    # Highlight peaks indicating periodicity
    peaks = np.where(autocorr > 0.7*np.max(autocorr))[0]
    plt.scatter(peaks, autocorr[peaks], color='red', s=40, 
                label='Significant Correlations')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot saved to {OUTPUT_PLOT}")
    plt.show()
