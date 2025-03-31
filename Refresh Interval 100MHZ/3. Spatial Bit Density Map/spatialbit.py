import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow

# Load video and initialize
video_path = r"C:\Users\aravi\100mhz dipswtich.mkv"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize density matrix
density_matrix = np.zeros((height, width), dtype=np.uint32)

# Track square positions and DIP events (example data - customize as needed)
dip_events = [
    {"start_time": 5, "end_time": 10, "label": "SW0 ON", "position": (100, 200)},
    {"start_time": 10, "end_time": 15, "label": "SW1 ON", "position": (300, 200)},
    {"start_time": 15, "end_time": 20, "label": "SW2 ON", "position": (500, 200)},
    # Add entries for SW3-SW9 based on your DIP timeline
]

# Process frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to binary (black square = 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
    density_matrix += binary.astype(np.uint32)
    frame_count += 1

cap.release()

# Normalize density
density_normalized = (density_matrix / frame_count * 255).astype(np.uint8)

# Create annotated plot
fig, ax = plt.subplots(figsize=(15, 8))
heatmap = ax.imshow(density_normalized, cmap='hot', interpolation='nearest')

# Add colorbar with time markers
cbar = fig.colorbar(heatmap, ax=ax)
cbar.set_label('Density of 1-Bits (White)', rotation=270, labelpad=20)

# Add DIP switch event annotations
for event in dip_events:
    x, y = event["position"]
    label = f"{event['label']}\n({event['start_time']}-{event['end_time']}s)"
    
    # Draw arrow and text
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + 50, y - 50),  # Adjust text position
        arrowprops=dict(facecolor='cyan', shrink=0.05),
        fontsize=10,
        color='white',
        bbox=dict(facecolor='black', alpha=0.7)
    )
    
    # Highlight position with a rectangle
    ax.add_patch(Rectangle(
        (x - 20, y - 20), 40, 40,  # Box around the square's position
        linewidth=2, edgecolor='lime', facecolor='none'
    ))

# Add timeline markers
time_per_pixel = (total_frames / fps) / width  # Time per pixel column
for sec in range(0, int(total_frames/fps) + 1, 5):
    x_pos = int(sec / time_per_pixel)
    ax.axvline(x=x_pos, color='blue', linestyle='--', alpha=0.5)
    ax.text(x_pos, -10, f"{sec}s", ha='center', color='blue')

# Customize plot
ax.set_title(f"Spatial Bit Density Heatmap (Video Duration: {total_frames/fps:.1f}s)")
ax.set_xlabel("X Position (Time markers at bottom)")
ax.set_ylabel("Y Position")
plt.tight_layout()
plt.savefig("annotated_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()
