# Mettler Toledo WXS Scale Communication Library (Simplified)
#
# This code provides a basic interface for serial communication with a Mettler WXS scale.
# It's a simplified version of the original HSL file, focusing on core functionality.
#
# Communication is based on sending simple text commands and parsing text responses.

import serial
import time


class MettlerWXS:
    """
    A class to handle serial communication with a Mettler Toledo WXS scale.
    """
    def __init__(self, port, baudrate=9600, timeout=1.0, simulating=False):
        """
        Initializes the MettlerWXS object and opens the serial port.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.simulating = simulating

        self._connect()

    def _connect(self):
        """
        Establishes the serial connection.
        """
        if self.simulating:
            print("Simulating Mettler WXS scale. No serial connection established.")
            return
        
        try:
            self.ser = serial.Serial(
                self.port,
                self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            print(f"Connected to {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"Error: Could not open serial port {self.port}. {e}")
            self.ser = None

    def _send_command(self, command, expected_response_prefix=None):
        """
        Sends a command to the scale and waits for a response.
        """

        if self.simulating:
            print(f"Simulating command: {command}")
            return "S S 0.000 g"  # Simulated response
        if not self.ser or not self.ser.is_open:
            print("Error: Serial port not connected.")
            return None

        # Add carriage return and line feed as per the original HSL code
        full_command = command + "\r\n"
        
        try:
            self.ser.write(full_command.encode('ascii'))
            time.sleep(0.1)  # Give the scale a moment to respond
            response = self.ser.readline().decode('ascii').strip()
            
            if expected_response_prefix and not response.startswith(expected_response_prefix):
                print(f"Warning: Unexpected response. Expected prefix '{expected_response_prefix}', got '{response}'.")

            return response
        except serial.SerialException as e:
            print(f"Communication error: {e}")
            return None

    def get_weight(self, immediately=False):
        """
        Gets the current weight from the scale.
        """
        command = "SI" if immediately else "S"
        response = self._send_command(command, "S ")
        
        if response and response.startswith("S S"):
            # Example response: S S 24.567 g
            parts = response.split()
            if len(parts) >= 3:
                try:
                    weight = float(parts[2])
                    unit = parts[3] if len(parts) > 3 else "Unknown"
                    return weight, unit
                except (ValueError, IndexError):
                    print("Error parsing weight response.")
        
        print(f"Failed to get stable weight. Response: {response}")
        return None, None

    def tare(self, immediately=False):
        """
        Tares the scale (sets the current weight to zero).
        """
        command = "TI" if immediately else "T"
        response = self._send_command(command, "T")
        
        if response and response.startswith(("T S", "TI S")):
            print("Scale successfully tared.")
            return True
        
        print(f"Failed to tare scale. Response: {response}")
        return False
        
    def close(self):
        """
        Closes the serial connection.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed.")
            self.ser = None

if __name__ == '__main__':
    # This is an example of how to use the MettlerWXS class.
    # Replace 'COM3' with the actual serial port your scale is connected to.
    
    # ⚠️ IMPORTANT: This will only work if you have the pyserial library installed.
    # You can install it with: pip install pyserial
    
    # It also requires a physical Mettler WXS scale connected to the specified COM port.
    
    scale = MettlerWXS(port='COM3')
    
    if scale.ser and scale.ser.is_open:
        weight, unit = scale.get_weight()
        if weight is not None:
            print(f"Current weight: {weight} {unit}")
        
        if scale.tare(immediately=True):
            time.sleep(1) # Give the scale a moment to settle
            weight, unit = scale.get_weight()
            if weight is not None:
                print(f"Weight after taring: {weight} {unit}")

        scale.close()