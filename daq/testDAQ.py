import nidaqmx

import time
import numpy as np

# Define your DAQ output channel
ao_channel_1 = "Dev1/ao1"  # Change to match your DAQ device
ao_channel_0 = "Dev1/ao0"  # Change to match your DAQ device

# Create the task
with nidaqmx.Task() as task:
    task.ao_channels.add_ao_voltage_chan(ao_channel_1)  # Add output channel
    task.ao_channels.add_ao_voltage_chan(ao_channel_0)  # Add output channel

    print("Starting real-time voltage output...")

    time_start = time.time()

    try:
        while time.time() - time_start < 5:
            # Get or compute the voltage value in real-time
            voltage_1 = 5*np.sin(time.time()*30)  # Example: a sine wave based on time
            voltage_2 = 0*np.sin(time.time()/1)  # Example: a sine wave based on time

            # Write the voltage to the output channel
            task.write([voltage_1, voltage_2])

            # Small delay to simulate real-time streaming (adjust as needed)
            # time.sleep(0.0001)  # 10 ms loop time

        while time.time() - time_start < 10:
            # Get or compute the voltage value in real-time
            voltage_1 = 5*np.sin(time.time()*30)  # Example: a sine wave based on time
            voltage_2 = 5*np.sin(time.time()/1)  # Example: a sine wave based on time

            # Write the voltage to the output channel
            task.write([voltage_1, voltage_2])

            # Small delay to simulate real-time streaming (adjust as needed)
            # time.sleep(0.0001)  # 10 ms loop time
        while True:
            # Get or compute the voltage value in real-time
            voltage_1 = 5*np.sin(time.time()*30)  # Example: a sine wave based on time
            voltage_2 = 5*np.sin(time.time()/1)  # Example: a sine wave based on time

            # Write the voltage to the output channel
            task.write([voltage_1, voltage_2])

            # Small delay to simulate real-time streaming (adjust as needed)
            # time.sleep(0.0001)  # 10 ms loop time
    except KeyboardInterrupt:
        print("Stopping voltage output.")