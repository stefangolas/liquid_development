# This code provides an interface for serial communication with a Mettler WXS scale.
# It has been enhanced to parse a wide range of status and error responses
# based on the provided HSL reference file.
#
# Communication is based on sending simple text commands and parsing text responses.

import serial
import time
import contextlib

class MettlerWXS:
    """
    A class to handle serial communication with a Mettler Toledo WXS scale.
    Implements a context manager to ensure the serial port is only open when needed.
    """
    # Nested class to hold all known response codes for clarity and maintenance
    class Responses:
        # Weight Responses
        WEIGHT_STABLE = "S S"
        WEIGHT_DYNAMIC = "S D"
        WEIGHT_ERROR = "S I"    # Command not executable
        WEIGHT_OVERLOAD = "S +"
        WEIGHT_UNDERLOAD = "S -"

        # Tare Responses
        TARE_STABLE = "T S"
        TARE_IMMEDIATE_STABLE = "TI S"
        TARE_DYNAMIC = "TI D"
        TARE_ERROR = "T I"
        TARE_IMMEDIATE_ERROR = "TI I"
        TARE_NOT_EXECUTABLE = "TI L"
        TARE_OVERLOAD = "T +"
        TARE_IMMEDIATE_OVERLOAD = "TI +"
        TARE_UNDERLOAD = "T -"
        TARE_IMMEDIATE_UNDERLOAD = "TI -"
        
        # Zero Responses
        ZERO_STABLE = "Z A"
        ZERO_IMMEDIATE_STABLE = "ZI S"
        ZERO_DYNAMIC = "Z D"
        ZERO_IMMEDIATE_DYNAMIC = "ZI D"
        ZERO_ERROR = "Z I"
        ZERO_IMMEDIATE_ERROR = "ZI I"
        ZERO_OVERLOAD = "Z +"
        ZERO_IMMEDIATE_OVERLOAD = "ZI +"
        ZERO_UNDERLOAD = "Z -"
        ZERO_IMMEDIATE_UNDERLOAD = "ZI -"
        
        # Command Execution Status
        CMD_SYNTAX_ERROR = "ES"
        CMD_EXECUTION_ERROR = "ET"
        CMD_OK = "EA"
        
    def __init__(self, port, baudrate=9600, timeout=1.0, simulating=False):
        """
        Initializes the MettlerWXS object, but DEFERRS opening the serial port
        until the context manager is entered.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.simulating = simulating
        # IMPORTANT CHANGE: Removed self._connect() here to prevent premature port locking.

    def __enter__(self):
        """Context manager entry point: establishes the serial connection."""
        print("Entering scale context: Attempting connection...")
        self._connect()
        if not self.ser and not self.simulating:
            # Raise an error if connection failed and we're not simulating
            raise ConnectionError(f"Failed to connect to Mettler scale on port {self.port}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point: ensures the serial connection is closed."""
        print("Exiting scale context: Closing connection...")
        self.close()
        return False # Propagate exceptions if they occurred inside the 'with' block

    def _connect(self):
        """
        Establishes the serial connection.
        """
        if self.ser and self.ser.is_open:
            print(f"Connection to {self.port} already open.")
            return

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
        except Exception as e:
            print(f"Unexpected connection error: {e}")
            self.ser = None


    def _send_command(self, command):
        """
        Sends a command to the scale and waits for a response.
        """
        if self.simulating:
            print(f"Simulating command: {command}")
            # Simulate different responses for testing
            if command in ["S", "SI"]:
                return self.Responses.WEIGHT_STABLE + " 123.45 g"
            if command in ["T", "TI"]:
                return self.Responses.TARE_STABLE
            if command in ["Z", "ZI"]:
                return self.Responses.ZERO_STABLE
            return "SIMULATED OK"

        if not self.ser or not self.ser.is_open:
            print("Error: Serial port not connected. Use context manager.")
            return None

        full_command = command + "\r\n"
        
        try:
            self.ser.flushInput()  # Clear input buffer before sending
            self.ser.write(full_command.encode('ascii'))
            time.sleep(0.1)  # Give the scale a moment to respond
            response = self.ser.readline().decode('ascii').strip()
            
            # Mettler scales sometimes send an echo or status before the actual reply.
            # This loop reads lines until it gets a meaningful response.
            while response == "" or response.startswith(" "):
                 response = self.ser.readline().decode('ascii').strip()

            print(f"Sent: {command} | Received: {response}")
            return response
        except serial.SerialException as e:
            print(f"Communication error: {e}")
            return None

    def get_weight(self, immediately=False):
        """
        Gets the current weight from the scale and parses the status.
        Returns a dictionary with status, value, and unit.
        """
        command = "SI" if immediately else "S"
        response = self._send_command(command)
        
        if not response:
            return {'status': 'no_response', 'value': None, 'unit': None}

        parts = response.split()
        status_code = " ".join(parts[0:2]) if len(parts) >= 2 else response

        # Create a dictionary to map response codes to status messages
        status_map = {
            self.Responses.WEIGHT_STABLE: 'stable',
            self.Responses.WEIGHT_DYNAMIC: 'dynamic',
            self.Responses.WEIGHT_ERROR: 'error_not_executable',
            self.Responses.WEIGHT_OVERLOAD: 'overload',
            self.Responses.WEIGHT_UNDERLOAD: 'underload',
            self.Responses.CMD_SYNTAX_ERROR: 'syntax_error'
        }

        status = status_map.get(status_code, 'unknown_response')
        result = {'status': status, 'value': None, 'unit': None, 'raw_response': response}

        if status == 'stable' and len(parts) >= 4:
            try:
                result['value'] = float(parts[2])
                result['unit'] = parts[3]
            except (ValueError, IndexError):
                result['status'] = 'parsing_error'
        
        return result

    def tare(self, immediately=False):
            """
            Tares the scale (sets the current weight to zero) and parses the response.
            Returns a dictionary with the operation status.
            """
            command = "TI" if immediately else "T"
            response = self._send_command(command)
            
            if not response:
                return {'status': 'no_response', 'raw_response': None}
                
            status_map = {
                # Check for immediate commands first, as they are more specific
                self.Responses.TARE_IMMEDIATE_STABLE: 'success',
                self.Responses.TARE_STABLE: 'success',
                self.Responses.TARE_DYNAMIC: 'dynamic_weight',
                self.Responses.TARE_NOT_EXECUTABLE: 'error_not_executable',
                self.Responses.TARE_IMMEDIATE_ERROR: 'error',
                self.Responses.TARE_ERROR: 'error',
                self.Responses.TARE_IMMEDIATE_OVERLOAD: 'overload',
                self.Responses.TARE_OVERLOAD: 'overload',
                self.Responses.TARE_IMMEDIATE_UNDERLOAD: 'underload',
                self.Responses.TARE_UNDERLOAD: 'underload'
            }
            
            status = 'unknown_response'
            for code, desc in status_map.items():
                if response.startswith(code):
                    status = desc
                    break  # Exit loop once a match is found
            
            return {'status': status, 'raw_response': response}

    def zero(self, immediately=False):
        """
        Zeros the scale and parses the response.
        Returns a dictionary with the operation status.
        """
        command = "ZI" if immediately else "Z"
        response = self._send_command(command)
        
        if not response:
            return {'status': 'no_response', 'raw_response': None}
            
        status_map = {
            self.Responses.ZERO_IMMEDIATE_STABLE: 'success',
            self.Responses.ZERO_STABLE: 'success',
            self.Responses.ZERO_IMMEDIATE_DYNAMIC: 'dynamic_weight',
            self.Responses.ZERO_DYNAMIC: 'dynamic_weight',
            self.Responses.ZERO_IMMEDIATE_ERROR: 'error',
            self.Responses.ZERO_ERROR: 'error',
            self.Responses.ZERO_IMMEDIATE_OVERLOAD: 'overload',
            self.Responses.ZERO_OVERLOAD: 'overload',
            self.Responses.ZERO_IMMEDIATE_UNDERLOAD: 'underload',
            self.Responses.ZERO_UNDERLOAD: 'underload'
        }
        
        status = 'unknown_response'
        for code, desc in status_map.items():
            if response.startswith(code):
                status = desc
                break # Exit loop once a match is found
        
        return {'status': status, 'raw_response': response}

    def close(self):
        """
        Closes the serial connection.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed.")
            self.ser = None

if __name__ == '__main__':
    # --- Example Usage using the new Context Manager ---
    #
    # The 'with' block ensures the connection is closed even if errors occur.
    # ⚠️ IMPORTANT: 
    #   - Install pyserial: pip install pyserial
    #   - Replace 'COM4' with your scale's actual serial port.
    #   - Set simulating=True to test without a physical scale.
    
    SCALE_PORT = 'COM4'
    SIMULATING = True
    
    try:
        with MettlerWXS(port=SCALE_PORT, simulating=SIMULATING) as scale:
        
            print("\n--- Getting Initial Weight ---")
            weight_data = scale.get_weight()
            print(f"Response: {weight_data}")
            if weight_data['status'] == 'stable':
                print(f"✅ Current weight: {weight_data['value']} {weight_data['unit']}")
            else:
                print(f"⚠️ Could not get a stable weight. Status: {weight_data['status']}")

            print("\n--- Taring the Scale ---")
            tare_result = scale.tare(immediately=True)
            print(f"Response: {tare_result}")
            if tare_result['status'] == 'success':
                print("✅ Scale successfully tared.")
                time.sleep(1)  # Give the scale a moment to settle

                print("\n--- Getting Weight After Taring ---")
                weight_after_tare = scale.get_weight()
                print(f"Response: {weight_after_tare}")
                if weight_after_tare['status'] == 'stable':
                    print(f"✅ Weight after taring: {weight_after_tare['value']} {weight_after_tare['unit']}")
                else:
                    print(f"⚠️ Could not get a stable weight. Status: {weight_after_tare['status']}")
            else:
                print(f"❌ Failed to tare scale. Status: {tare_result['status']}")

    except ConnectionError as e:
        print(f"\n❌ Connection failed: {e}")
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")