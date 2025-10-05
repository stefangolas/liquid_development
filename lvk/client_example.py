import requests
import json
from typing import List, Dict, Union, Optional
import numpy as np
import itertools
from typing import Dict, List, Union, Tuple


class LiquidClassClient:
    """
    Programmatic client for the Liquid Class Development API.

    Inferred API Endpoints:
    1. GET /search_liquid_classes?q={query}
    2. GET /get_liquid_class/{name}
    3. POST /initialize_hamilton
    4. POST /execute_pipette (with liquid class data)
    """

    # The actual base URL would need to be known, but we'll use a placeholder.
    # The client assumes the API is running on a server accessible at this base_url.
    # Example: 'http://localhost:5000'
    def __init__(self, base_url: str):
        self.base_url = base_url
        print(f"Client initialized with Base URL: {self.base_url}")


    # --- 2. Get Liquid Class Details (GET) ---
    def get_liquid_class(self, class_name: str) -> Optional[Dict[str, Union[float, int, List[float]]]]:
        """
        Fetches the full parameter set for a specific liquid class.
        Endpoint: /get_liquid_class/{name}

        :param class_name: The exact name of the liquid class.
        :return: A dictionary of parameters or None on failure.
        """
        endpoint = f"{self.base_url}/get_liquid_class/{class_name}"
        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching liquid class '{class_name}': {e}")
            return None

    # --- 3. Execute Test Actions (POST) ---

    def initialize_hamilton(self) -> str:
        """
        Sends a POST request to initialize the instrument.
        Endpoint: /initialize_hamilton

        :return: Status message.
        """
        endpoint = f"{self.base_url}/initialize_hamilton"
        try:
            # The client-side JS sends an empty JSON body
            response = requests.post(endpoint, headers={'Content-Type': 'application/json'}, data=json.dumps({}), timeout=30)
            response.raise_for_status()
            return "Initialization complete."
        except requests.exceptions.RequestException as e:
            return f"Initialization failed: {e}"

    def import_liquid_class(self, liquid_class_data: Dict) -> Dict[str, Union[bool, str]]:
        """
        Imports a liquid class into the Hamilton robot.
        Endpoint: /import_liquid_class

        :param liquid_class_data: Dictionary containing the liquid class parameters 
                                  (aspirate, dispense, tip_type, dispense_mode, correction_curve).
        :return: The server's response (e.g., success status and a message).
        """

        endpoint = f"{self.base_url}/import_liquid_class"
        payload = {
            "liquid_class_dictionary": [liquid_class_data]
        }

        try:
            response = requests.post(
                endpoint,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Import failed: {e}"}

    def execute_pipette(self, liquid_class_name: str, volume: float) -> Dict[str, Union[bool, str]]:
        """
        Executes a pipette action with the specified liquid class parameters and volume.
        Endpoint: /execute_pipette

        :param liquid_class_data: Dictionary containing the liquid class parameters 
                                  (aspirate, dispense, tip_type, dispense_mode, correction_curve).
        :param volume: The volume (uL) for the test action.
        :return: The server's response (e.g., success status and a message).
        """


        endpoint = f"{self.base_url}/pipette_and_record"
        
        
        payload = {
            "liquid_class": liquid_class_name,
            "volume": volume
        }

        try:
            response = requests.post(
                endpoint,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Pipette execution failed: {e}"}
        
    def import_and_test_pipette(self, liquid_class_data: Dict, volume: float) -> Dict[str, Union[bool, str, Dict]]:
        """
        Imports a liquid class and immediately executes a pipette and weigh cycle.
        Endpoint: /import_and_test_liquid_class

        :param liquid_class_data: Dictionary containing the liquid class parameters.
        :param volume: The volume (uL) for the test action.
        :return: The server's response (e.g., success status, weight, TADM data).
        """
        endpoint = f"{self.base_url}/import_and_test_liquid_class"
        
        # The payload must now contain BOTH the liquid class dictionary (as a list) and the volume
        payload = {
            "liquid_class_dictionary": [liquid_class_data], # The server expects a list
            "volume": volume
        }

        print(f"Sending combined import and test for liquid class '{liquid_class_data['name']}' at {volume} uL...")
        response = requests.post(
            endpoint,
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=120  # Increased timeout for a combined, long-running robot task
        )
        response.raise_for_status()
        return response.json()




def pipette_using_parameters(lvk_client:LiquidClassClient, liquid_class_data: Dict, volume: float) -> Dict[str, Union[bool, str]]:
    response = lvk_client.import_liquid_class(liquid_class_data)
    print(response)
    response = lvk_client.execute_pipette(liquid_class_data['name'], volume)
    print(response)
    return response



starting_params ={
    "name": "beepboop2",
    "aspirate": {
        "FLOW_RATE": 10,
        "MIX_FLOW_RATE": 75,
        "AIR_TRANSPORT_VOLUME": 1,
        "BLOW_OUT_VOLUME": 3,
        "SWAP_SPEED": 1,
        "SETTLING_TIME": 1,
        "OVER_ASPIRATE_VOLUME": 0,
        "CLOT_RETRACT_HEIGHT": 0
    },
    "dispense": {
        "FLOW_RATE": 10,
        "MIX_FLOW_RATE": 75,
        "AIR_TRANSPORT_VOLUME": 1,
        "BLOW_OUT_VOLUME": 3,
        "SWAP_SPEED": 4,
        "SETTLING_TIME": 2,
        "STOP_FLOW_RATE": 1,
        "STOP_BACK_VOLUME": 0
    },
    "tip_type": {
        "volume": 50,
        "has_filter": False
    },
    "dispense_mode": "Surface Empty",
    "correction_curve": {
        "nominal": [
            0,
            15,
            20,
            30,
            50
        ],
        "corrected": [
            0,
            16,
            22,
            33.1,
            55
        ]
    }
}

import json
from typing import List, Dict, Union, Optional
from datetime import datetime

class LCOptimization:
    def __init__(self, lvk_client: LiquidClassClient, starting_params: Dict, density: float, history_file: str = "test_history.json"):
        self.lvk_client = lvk_client
        self.starting_params = starting_params
        self.density = density
        self.test_history: List[Dict] = []
        self.history_file = history_file
        
        # Load existing history if available
        try:
            with open(self.history_file, 'r') as f:
                self.test_history = json.load(f)
            print(f"Loaded {len(self.test_history)} previous test records from {self.history_file}")
        except FileNotFoundError:
            print(f"No existing history file found, starting fresh: {self.history_file}")

    def _append_to_history_file(self, record: Dict):
        """Append a single test record to both memory and JSON file."""
        self.test_history.append(record)
        with open(self.history_file, 'w') as f:
            json.dump(self.test_history, f, indent=4)

    def test_parameters(self, volume: float) -> float:
        """Tests the current starting_params and immediately appends the result to file."""
        response = self.lvk_client.import_and_test_pipette(self.starting_params, volume)
        timestamp = datetime.now().isoformat()
        test_record = {
            "timestamp": timestamp,
            "liquid_class_name": self.starting_params.get("name", "unknown"),
            "parameters": {
                "aspirate": self.starting_params.get("aspirate", {}).copy(),
                "dispense": self.starting_params.get("dispense", {}).copy(),
                "tip_type": self.starting_params.get("tip_type", {}).copy(),
                "dispense_mode": self.starting_params.get("dispense_mode", ""),
                "correction_curve": self.starting_params.get("correction_curve", {}).copy()
            },
            "volume_uL": volume
        }

        if response.get("status") != "success":
            print(f"Error during pipetting: {response.get('message')}")
            test_record.update({
                "expected_mass_g": (volume / 1000) * self.density,
                "measured_mass_g": float('nan'),
                "weight_status": "error",
                "tadm_data": None,
                "error_message": response.get("message", "Unknown error"),
                "test_status": "failed"
            })
            self._append_to_history_file(test_record)
            return float('nan')

        # Extract results
        test_result = response.get("test_result", {})
        weight_data = test_result.get("weight", {})
        weight_value = weight_data.get("value", float('nan'))
        weight_status = weight_data.get("status", "unknown")
        tadm_data = test_result.get("tadm_data", {})

        expected_mass = (volume / 1000) * self.density
        accuracy_percent = None
        if weight_value and expected_mass and not (isinstance(weight_value, float) and weight_value != weight_value):
            accuracy_percent = (weight_value / expected_mass) * 100 if expected_mass != 0 else None

        test_record.update({
            "expected_mass_g": expected_mass,
            "measured_mass_g": weight_value,
            "accuracy_percent": accuracy_percent,
            "weight_status": weight_status,
            "tadm_data": tadm_data,
            "test_status": "success"
        })

        self._append_to_history_file(test_record)

        print(f"Expected mass: {expected_mass:.6f} g, Measured mass: {weight_value:.6f} g")
        if accuracy_percent:
            print(f"Accuracy: {accuracy_percent:.2f}%")

        return weight_value

    # --- Utility Methods ---
    def get_test_history(self) -> List[Dict]:
        return self.test_history

    def save_test_history(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.test_history, f, indent=4)
        print(f"Test history saved to {filepath}")

    def get_latest_test(self) -> Optional[Dict]:
        return self.test_history[-1] if self.test_history else None

    def print_test_summary(self):
        if not self.test_history:
            print("No tests have been performed yet.")
            return
        
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY - Total Tests: {len(self.test_history)}")
        print(f"{'='*80}")
        
        for i, test in enumerate(self.test_history, 1):
            print(f"\nTest #{i} - {test['timestamp']}")
            print(f"  Liquid Class: {test['liquid_class_name']}")
            print(f"  Volume: {test['volume_uL']} uL")
            print(f"  Expected Mass: {test.get('expected_mass_g', float('nan')):.6f} g")
            print(f"  Measured Mass: {test.get('measured_mass_g', float('nan')):.6f} g")
            if test.get('accuracy_percent'):
                print(f"  Accuracy: {test['accuracy_percent']:.2f}%")
            print(f"  Status: {test['test_status']}")
            if test.get('error_message'):
                print(f"  Error: {test['error_message']}")

    # --- Grid Testing ---
    def test_parameter_grid(
        self,
        volume: float,
        param_spaces: Dict[str, Dict[str, Union[float, List[float]]]],
        samples_per_combination: int = 1,
        linked_params: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict]:
        """
        Iteratively tests a grid of parameter combinations, optionally repeating each combination.
        Also allows linking parameters across sections, e.g., blowout_volume should be same for aspirate & dispense.

        Args:
            volume (float): Volume to pipette in uL.
            param_spaces (Dict[str, Dict[str, float]]): Dictionary specifying parameters to sweep.
            samples_per_combination (int): Number of repeats per combination.
            linked_params (Dict[str, List[str]]): Dict mapping a canonical param name to list of sections it should be synced across.
                Example: {'BLOW_OUT_VOLUME': ['aspirate', 'dispense'], 'AIR_TRANSPORT_VOLUME': ['aspirate', 'dispense']}
        
        Returns:
            List[Dict]: Full test history.
        """
        # Build sweep lists
        sweep_lists = {}
        for section, params in param_spaces.items():
            sweep_lists[section] = {}
            for param_name, values in params.items():
                if not isinstance(values, list):
                    values = [values]
                sweep_lists[section][param_name] = values

        # Generate all combinations
        sections = list(sweep_lists.keys())
        param_names_per_section = {sec: list(sweep_lists[sec].keys()) for sec in sections}
        
        # Create a flat list of (section, param_name, value) tuples for all combinations
        all_params = []
        for sec in sections:
            for param_name in param_names_per_section[sec]:
                all_params.append((sec, param_name, sweep_lists[sec][param_name]))
        
        # Generate all combinations of values
        param_keys = [(sec, param) for sec, param, _ in all_params]
        value_lists = [values for _, _, values in all_params]
        full_combinations = list(itertools.product(*value_lists))

        original_params = {sec: self.starting_params[sec].copy() for sec in sections}
        linked_params = linked_params or {}

        for combination in full_combinations:
            # Set the values in starting_params
            for i, (sec, param_name) in enumerate(param_keys):
                self.starting_params[sec][param_name] = float(combination[i])

            # Apply linked parameters - take value from whichever section is in the grid
            for param_name, sec_list in linked_params.items():
                # Find which section has this parameter in the grid
                source_value = None
                for sec in sec_list:
                    if sec in param_names_per_section and param_name in param_names_per_section[sec]:
                        source_value = self.starting_params[sec][param_name]
                        break
                
                # If we found a value from the grid, apply it to all linked sections
                if source_value is not None:
                    for sec in sec_list:
                        if sec in self.starting_params:
                            self.starting_params[sec][param_name] = source_value

            # Run the specified number of repeats
            for sample_idx in range(samples_per_combination):
                print(f"\n--- Testing combination: { {sec:self.starting_params[sec] for sec in sections} }, Sample {sample_idx+1}/{samples_per_combination} ---")
                self.test_parameters(volume)

        # Restore original parameters
        for sec in sections:
            self.starting_params[sec] = original_params[sec]

        return self.get_test_history()


def generate_log_param_grid(
    param_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    points_per_param: int = 5
) -> Dict[str, Dict[str, List[float]]]:
    """
    Generates a log-spaced parameter grid suitable for `test_parameter_grid`.
    """
    log_grid: Dict[str, Dict[str, List[float]]] = {}
    
    for section, params in param_ranges.items():
        log_grid[section] = {}
        for param_name, (min_val, max_val) in params.items():
            if min_val <= 0:
                raise ValueError(f"Log scale requires min_val > 0 for parameter '{param_name}'")
            # Generate log-spaced points
            values = np.logspace(np.log10(min_val), np.log10(max_val), points_per_param)
            log_grid[section][param_name] = values.tolist()
    
    return log_grid


if __name__ == "__main__":
    client = LiquidClassClient("http://localhost:5000")
    
    # 1. Initialize the Hamilton (Still a separate, required step)
    init_status = client.initialize_hamilton()
    print(f"Hamilton Initialization Status: {init_status}")

    optimizer = LCOptimization(client, starting_params, density=1.0)
    param_ranges = {
        'dispense': {
            'BLOW_OUT_VOLUME': (1, 20)
        },
    }

    param_spaces = generate_log_param_grid(param_ranges, points_per_param=3)

    linked = {
        "BLOW_OUT_VOLUME": ["aspirate", "dispense"],
        "AIR_TRANSPORT_VOLUME": ["aspirate", "dispense"]
    }

    history = optimizer.test_parameter_grid(volume=2, param_spaces=param_spaces, samples_per_combination=3, linked_params=linked)

    optimizer.print_test_summary()








