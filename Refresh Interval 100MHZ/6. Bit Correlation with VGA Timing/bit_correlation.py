import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================
# CONFIGURATION
# ========================
VIDEO_PATH = r"C:\Users\aravi\100mhz dipswtich.mkv"
HSYNC_THRESH = 0.95  # 95% black pixels for horizontal sync detection
VSYNC_THRESH = 0.95  # 95% black pixels for vertical sync detection
FRAME_SKIP = 1       # Process every frame (increase for faster processing)

# ========================
# VGA TIMING CONSTANTS (Modify for your resolution)
# ========================
EXPECTED_HSYNC_PER_FRAME = 525   # 480p: 525 lines/frame
EXPECTED_VSYNC_PER_SEC = 60       # 60Hz refresh rate

# ========================
# MAIN PROCESSING
# ========================
def detect_sync_pulses(frame):
    """Detect HSYNC and VSYNC in a frame using pixel analysis"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
    
    # Horizontal sync detection (scan last 5% rows)
    hsync_zone = binary[-int(binary.shape[0]*0.05):, :]
    hsync_lines = np.where(hsync_zone.mean(axis=1) > HSYNC_THRESH)[0]
    
    # Vertical sync detection (scan last 5% columns)
    vsync_zone = binary[:, -int(binary.shape[1]*0.05):]
    vsync_cols = np.where(vsync_zone.mean(axis=0) > VSYNC_THRESH)[0]
    
    return len(hsync_lines), len(vsync_cols)

def analyze_vga_timing():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    data = {'frame': [], 'hsync': [], 'vsync': [], 'bit1': [], 'bit0': []}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % FRAME_SKIP == 0:
            # Detect sync pulses
            hsync_count, vsync_count = detect_sync_pulses(frame)
            
            # Count bits
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
            bit1 = binary.sum()
            bit0 = binary.size - bit1
            
            # Store data
            data['frame'].append(frame_idx)
            data['hsync'].append(hsync_count)
            data['vsync'].append(vsync_count)
            data['bit1'].append(bit1)
            data['bit0'].append(bit0)
        
        frame_idx += 1
    
    cap.release()
    return data

# ========================
# ANALYSIS & PLOTTING
# ========================
def plot_sync_correlation(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # HSYNC vs Bit Count
    ax1.scatter(data['hsync'], data['bit1'], c='blue', label='1 Bits')
    ax1.scatter(data['hsync'], data['bit0'], c='red', label='0 Bits')
    ax1.axvline(EXPECTED_HSYNC_PER_FRAME, color='black', linestyle='--', 
               label=f'Expected ({EXPECTED_HSYNC_PER_FRAME})')
    ax1.set_title('Horizontal Sync Correlation')
    ax1.set_xlabel('HSYNC Pulses per Frame')
    ax1.set_ylabel('Bit Count')
    ax1.legend()
    ax1.grid(True)

    # VSYNC vs Bit Count
    ax2.scatter(data['vsync'], data['bit1'], c='blue', label='1 Bits')
    ax2.scatter(data['vsync'], data['bit0'], c='red', label='0 Bits')
    ax2.axvline(EXPECTED_VSYNC_PER_SEC, color='black', linestyle='--',
               label=f'Expected ({EXPECTED_VSYNC_PER_SEC}Hz)')
    ax2.set_title('Vertical Sync Correlation')
    ax2.set_xlabel('VSYNC Pulses per Second')
    ax2.set_ylabel('Bit Count')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('vga_sync_correlation.png', dpi=300)
    plt.show()

# ========================
# EXECUTION
# ========================
if __name__ == "__main__":
    print("Analyzing VGA timing correlation...")
    timing_data = analyze_vga_timing()
    print("Generating diagnostic plots...")
    plot_sync_correlation(timing_data)
    print("Analysis complete. Check vga_sync_correlation.png")
