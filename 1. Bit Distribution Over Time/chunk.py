import csv
import os

file_path = r"C:\Users\aravi\input.mp4.mkv"  # Replace with your path
output_csv = "bit_distribution.csv"

# Initialize counters
bit_data = []
bytes_processed = 0
chunk_size = 1024  # Adjust for granularity (smaller = more data points)

try:
    with open(file_path, "rb") as file:
        print("Processing...")
        while chunk := file.read(chunk_size):
            ones = sum(bin(byte).count('1') for byte in chunk)
            zeros = len(chunk) * 8 - ones  # Total bits = bytes * 8
            
            # Track time: (bytes_processed / total_file_size) * video_duration (if known)
            bit_data.append([bytes_processed, ones, zeros])
            bytes_processed += len(chunk)
    
    # Save to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Byte_Position", "Bit_1_Count", "Bit_0_Count"])
        writer.writerows(bit_data)
    
    print(f"Data saved to {output_csv}")

except FileNotFoundError:
    print("File not found. Verify path:", file_path)
except Exception as e:
    print(f"Error: {e}")
