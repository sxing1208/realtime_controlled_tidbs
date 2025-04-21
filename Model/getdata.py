import serial
import time
import sys
import numpy as np

filename = "static"

# Configure the serial port
ser = serial.Serial('COM4', 115200)  # Replace 'COM3' with your serial port
data_list = []
start_time = None  # To store the timestamp of the first entry

try:
    print("Reading data from serial port. Press Ctrl+C to stop...")
    while True:
        # Read a line from the serial port
        line = ser.readline().decode('utf-8').strip()
        
        # Split the line into x, y, z components
        try:
            x, y, z = map(float, line.split(','))
            
            # Get the current timestamp in seconds
            current_time = time.time()
            
            # If this is the first entry, set it as the reference time (zero)
            if start_time is None:
                start_time = current_time
                timestamp = 0.0  # First entry has timestamp zero
            else:
                # Calculate the time delta relative to the first entry
                timestamp = current_time - start_time
            
            # Append the data to the list
            data_list.append((timestamp, x, y, z))
            
            # Print the data for debugging
            print(f"Timestamp: {timestamp:.6f}, X: {x}, Y: {y}, Z: {z}")
        except ValueError:
            print(f"Invalid data format: {line}")

except KeyboardInterrupt:
    print("\nData collection stopped by user.")

finally:
    # Close the serial port
    ser.close()
    
    # Convert the list to a 4xT numpy array
    if data_list:
        data_array = np.array(data_list).T
        print("\nData array shape:", data_array.shape)
        print("Data array:")
        print(data_array)
        np.save(filename, data_array)
    else:
        print("No data collected.")