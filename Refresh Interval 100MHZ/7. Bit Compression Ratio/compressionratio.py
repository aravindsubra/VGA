# Install required system libraries first (Linux)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
VIDEO_PATH = r'C:\Users\aravi\100mhz dipswtich.mkv'  # Relative path recommended
FRAME_SKIP = 5  # Process every 5th frame for efficiency

def simple_compress(data):
    """Run-length encoding compression example"""
    compressed = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            compressed.append((data[i - 1], count))
            count = 1
    compressed.append((data[-1], count))
    return len(compressed) * 2  # Estimate storage size (value + count)

# Initialize video processing
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video file: {VIDEO_PATH}")

original_bit_counts = []
compressed_sizes = []

# Process video frames
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % FRAME_SKIP == 0:
        # Convert to binary (black square = 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
        
        # Calculate metrics
        bit1_count = binary.sum()
        compressed_size = simple_compress(binary.flatten())
        
        original_bit_counts.append(bit1_count)
        compressed_sizes.append(compressed_size)
    
    frame_idx += 1

cap.release()

# Create and save DataFrame
df = pd.DataFrame({
    'Frame': range(len(original_bit_counts)),
    'Original_Bits': original_bit_counts,
    'Compressed_Size': compressed_sizes
})
df['Compression_Ratio'] = df['Original_Bits'] / df['Compressed_Size']
df.to_csv('compression_ratios.csv', index=False)

# Generate plot
plt.figure(figsize=(12, 6))
plt.bar(df['Frame'], df['Compression_Ratio'], color='purple', alpha=0.7)
plt.title('Frame Compression Ratios\n(Static Patterns = Higher Ratios)')
plt.xlabel('Frame Number (Every 5th Frame)')
plt.ylabel('Compression Ratio (Original/Compressed)')
plt.grid(axis='y', linestyle='--')

# Highlight high-compression frames
threshold = df['Compression_Ratio'].quantile(0.9)
high_compression = df[df['Compression_Ratio'] > threshold]
plt.scatter(high_compression['Frame'], high_compression['Compression_Ratio'], 
            color='red', zorder=3, label='High Redundancy Frames')

plt.legend()
plt.tight_layout()
plt.savefig('compression_analysis.png', dpi=300)
plt.show()
