import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# Configuration
VIDEO_PATH = r"C:\Users\aravi\bit_rate vs dip_switch.mkv"
DIP_TIMELINE = [
    (0, 5, "ALL_OFF_1"),
    (5, 10, "SW0_ON"),
    (10, 15, "SW1_ON"),
    (15, 20, "SW2_ON"),
    (20, 25, "SW3_ON"),
    (25, 30, "SW4_ON"),
    (30, 35, "SW5_ON"),
    (35, 40, "SW6_ON"),
    (40, 45, "SW7_ON"),
    (45, 50, "SW8_ON"),
    (50, 55, "SW9_ON"),
    (55, 60, "ALL_OFF_2")
]

def process_segment(args):
    """Process video segment using multiprocessing"""
    start, end, dip = args
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    
    total_ones = 0
    total_bits = 0
    frame_count = 0
    
    while cap.isOpened() and (frame_count / fps) < (end - start):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame (5x speedup)
        if frame_count % 5 != 0:
            frame_count += 1
            continue
            
        # Fast grayscale conversion and downscaling
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
        
        # Ultra-fast bit counting using NumPy
        bits = np.unpackbits(gray)
        total_ones += np.sum(bits)
        total_bits += bits.size
        
        frame_count += 1
    
    cap.release()
    return dip, total_ones, total_bits - total_ones

if __name__ == "__main__":
    # Parallel processing
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_segment, DIP_TIMELINE)
    
    # Create DataFrame
    df = pd.DataFrame(results, columns=["DIP_Config", "Bit1", "Bit0"])
    df["1/0 Ratio"] = df["Bit1"] / df["Bit0"]
    
    # Save results
    df.to_csv("dip_analysis_optimized.csv", index=False)
    
    # Generate plot
    plt.figure(figsize=(14, 7))
    bars = plt.bar(df["DIP_Config"], df["1/0 Ratio"], color='darkorange')
    plt.title("Bit Ratio vs. DIP Switch Configuration (Optimized)")
    plt.xlabel("DIP Switch State", fontsize=12)
    plt.ylabel("1/0 Ratio", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig("optimized_dip_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
