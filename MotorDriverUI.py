import tkinter as tk
import serial
import time
import threading

"""
Setup:
- Connect switch signal and ground input to Arduino port 13 and GND, respectively
- Upload Solenoid_Motor_Driver.ino to connected Arduino
- Replace the port in init_serial() to port corresponding to arduino connection
- Run this file, if set-up is correct Simple UI will pop up.
"""

# Initialize serial connection with Arduino
def init_serial():
    try:
        ser = serial.Serial('COM7', 9600, timeout=1)  # Update 'COM7' to your Arduino port
        time.sleep(2)  # Give some time for the connection to establish
        return ser
    except Exception as e:
        print("Error: Could not connect to Arduino")
        print(e)
        return None

# Send frequency to Arduino
def set_frequency():
    try:
        frequency = int(frequency_input.get())  # Get the frequency from the GUI input
        ser.write(f'{frequency},'.encode())    # Send the frequency over serial
        print(f"Sent frequency: {frequency}")
    except ValueError:
        print("Please enter a valid frequency")

# Trigger the second task in Arduino
def trigger_frequency_sweep():
    try:
        ser.write('-1,'.encode())  # Send command to trigger second task (simulating separate thread)
        print("Triggered second task")
    except:
        print("Error: Could not send command")

# Stop any ongoing function in Arduino
def stop_all_operations():
    try:
        ser.write('-2,'.encode())  # Send command to trigger second task (simulating separate thread)
        print("Triggered stop command")
    except:
        print("Error: Could not send command")

# Read serial logs from Arduino
def read_serial_logs():
    while True:
        if ser.in_waiting:  # Check if data is available to read
            line = ser.readline().decode('utf-8').strip()  # Read a line from the serial
            print(f"Received: {line}")  # Print the received line
        time.sleep(0.1)  # Small delay to avoid CPU overloading

# Create the main application window
root = tk.Tk()
root.title("Vibration Frequency Controller")

# Frequency input label and entry
tk.Label(root, text="Enter Frequency (Hz):").pack(pady=10)
frequency_input = tk.Entry(root)
frequency_input.pack(pady=10)

# Button to send the frequency
send_button = tk.Button(root, text="Set Frequency", command=set_frequency)
send_button.pack(pady=10)

# Button to trigger second task
frequency_sweep_button = tk.Button(root, text="Trigger Frequency Sweep", command=trigger_frequency_sweep)
frequency_sweep_button.pack(pady=10)

# Button to stop
frequency_sweep_button = tk.Button(root, text="Cancel", command=stop_all_operations)
frequency_sweep_button.pack(pady=10)


# Initialize serial connection with Arduino
ser = init_serial()

# Start a thread to read serial logs
log_thread = threading.Thread(target=read_serial_logs, daemon=True)
log_thread.start()

# Run the application
root.mainloop()

# Close serial connection when the GUI is closed
if ser:
    ser.close()
