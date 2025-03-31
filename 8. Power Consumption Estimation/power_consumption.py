import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ========================
# CONFIGURATION (ADJUST THESE)
# ========================
VIDEO_PATH = r"C:\Users\aravi\bit_rate vs dip_switch.mkv"
FRAME_SKIP = 10                # Process every 10th frame
VOLTAGE = 1.2                  # FPGA core voltage (adjust for your board)
POWER_PER_BIT = 0.000001       # Watts per 1-bit (typical FPGA ~1μW/bit)
DOWNSCALE_FACTOR = 4           # Reduce resolution for faster processing
DIP_TIMELINE = [               # Your DIP switch activation times
    (0, 5, "ALL_OFF"),
    (5, 10, "SW0_ON"),
    (10, 15, "SW1_ON")
]

# ========================
# POWER ANALYSIS FUNCTIONS
# ========================
def estimate_power(frame):
    """Calculate power consumption for a frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0,0), fx=1/DOWNSCALE_FACTOR, fy=1/DOWNSCALE_FACTOR)
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
    return binary.sum() * POWER_PER_BIT * VOLTAGE

# ========================
# MAIN PROCESSING
# ========================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
timestamps = []
power_values = []
dip_labels = []

print("Processing video for power estimation...")
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Skip frames for speed
    if frame_count % FRAME_SKIP != 0:
        frame_count += 1
        continue
    
    # Calculate power
    power = estimate_power(frame)
    current_time = frame_count / fps
    
    # Track DIP switch states
    current_dip = "UNKNOWN"
    for start, end, label in DIP_TIMELINE:
        if start <= current_time < end:
            current_dip = label
            break
    
    timestamps.append(current_time)
    power_values.append(power)
    dip_labels.append(current_dip)
    
    frame_count += 1

cap.release()

# ========================
# VISUALIZATION
# ========================
plt.figure(figsize=(14, 6))
plt.plot(timestamps, power_values, 'b-', label='Power Consumption')

# Add DIP switch annotations
unique_dips = list(set(dip_labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_dips)))
for dip, color in zip(unique_dips, colors):
    indices = [i for i, label in enumerate(dip_labels) if label == dip]
    if indices:
        plt.scatter([timestamps[i] for i in indices], 
                   [power_values[i] for i in indices],
                   color=color, label=dip, s=40)

plt.title(f"FPGA Power Estimation\n({VIDEO_PATH})")
plt.xlabel("Time (seconds)")
plt.ylabel(f"Power (W)\n(V={VOLTAGE}V, {POWER_PER_BIT*1e6}μW/bit)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save outputs
plt.savefig('power_consumption_plot.png', dpi=300)
pd.DataFrame({
    'Timestamp': timestamps,
    'Power(W)': power_values,
    'DIP_Config': dip_labels
}).to_csv('power_consumption_data.csv', index=False)

print("Power analysis complete. Outputs saved:")
print("- power_consumption_plot.png")
print("- power_consumption_data.csv")
