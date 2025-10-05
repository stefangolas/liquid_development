# Assuming these imports are necessary from your project's environment
from pyhamilton import (HamiltonInterface, LayoutManager,
                        layout_item, ReagentTrackedBulkPlate,
                        Tip96, TrackedTips, LVKBalanceVial, normal_logging,
                        create_liquid_class_from_json, create_liquid_class_from_dict)
from pyhamilton.pipetting import pip_transfer
from mettler_toledo import MettlerWXS
import os
import time


class LVKController:
    """
    Manages and controls a Hamilton pipetting robot with a Mettler scale
    for automated liquid handling and weighing. This class is designed to be
    used as a backend for API calls.
    """

    def __init__(self, layout_file, scale_com_port, simulating=True):
        """
        Initializes the hardware and layout configuration.

        Args:
            layout_file (str): Path to the Hamilton layout file (.lay).
            scale_com_port (str): The COM port for the Mettler scale (e.g., 'COM4').
            simulating (bool): If True, run in simulation mode without real hardware.
        """
        self.simulating = simulating
        self.scale_com_port = scale_com_port

        # 1. Load layout and define deck resources
        print("Initializing layout manager...")
        self.lmgr = LayoutManager(layout_file)

        self.lvk_vial = layout_item(self.lmgr, LVKBalanceVial, 'LVK_BALANCE_VIAL_0001')
        self.lvk_vial_position = [(self.lvk_vial, 0)]

        self.source_trough = layout_item(self.lmgr, ReagentTrackedBulkPlate, 'rgt_cont_60ml_BC_A00_0001')
        self.source_position = [(self.source_trough, 0)]


        # 3. Configure Pipette Tips
        print("Configuring pipette tips...")
        self.tips = TrackedTips.from_prefix(
            tracker_id="TIP_50uLF_L", volume_capacity=50, prefix="TIP_50uLF_L",
            count=8, tip_type=Tip96, lmgr=self.lmgr
        )
        print("Hardware configuration loaded. Ready to connect to the robot.")

    def connect(self):
        """
        Establishes and initializes the connection to the Hamilton robot.
        This should be called once before performing pipetting actions.
        """

        print("Connecting to Hamilton interface...")
        # persistent=True is ideal for a backend API to keep the connection alive
        with HamiltonInterface(windowed=True, simulating=False, persistent=True) as ham_int:
            ham_int.initialize()
            print("Hamilton robot connection initialized successfully.")

    def disconnect(self):
        """Closes the connection to the Hamilton robot."""
        with HamiltonInterface(windowed=True, simulating=False, persistent=True) as ham_int:
            ham_int.stop()
            print("Hamilton robot connection stopped successfully.")


    def tare_scale(self):
        """Tares the Mettler scale to zero and confirms."""
        print("Taring scale...")
        # USE THE CONTEXT MANAGER
        print("Opening scale connection...")
        print(self.scale_com_port)
        print(f"Simulating: {self.simulating}")
        with MettlerWXS(self.scale_com_port, simulating=False) as scale:
            scale.tare(immediately=True)
        print("Scale tared.")

    def get_scale_weight(self):
        """
        Retrieves the current weight from the Mettler scale.
        ...
        """
        # USE THE CONTEXT MANAGER
        with MettlerWXS(self.scale_com_port, simulating=False) as scale:
            while True:
                weight = scale.get_weight(immediately=True)
                if weight['status'] != 'stable':
                    print("Weight unstable, retrying...")
                    time.sleep(0.5)
                    continue
                else:
                    print(f"Weight measured: {weight['value']} g")
                    return weight
        
    def import_liquid_class_from_dictionary(self, liquid_class_dict):
        with HamiltonInterface(windowed=True, simulating=False, persistent=True) as ham_int:
            print("Liquid class dictionary")
            print(liquid_class_dict)
            create_liquid_class_from_dict(ham_int, liquid_class_dict) 
            print(f"Liquid class imported from dictionary successfully.")

    def pipette_and_weigh(self, volume, liquid_class):
        """
        Performs a full pipetting and weighing cycle.

        Args:
            volume (float): The volume to pipette in microliters.
            liquid_class (str): The name of the liquid class to use for the transfer.

        Returns:
            float: The final weight recorded by the scale after dispensing.
        """
        with HamiltonInterface(windowed=True, simulating=False, persistent=True) as ham_int:
            print(f"Starting pipetting cycle: {volume}uL with liquid class '{liquid_class}'")
            self.tare_scale()

            print(f"Aspirating and dispensing {volume}uL...")
            pip_transfer(
                ham_int,
                self.tips,
                self.source_position,
                self.lvk_vial_position,
                volumes=[volume],
                liquid_class=liquid_class,
                dispense_height=1,
            )
            print("Pipetting transfer complete.")

            weight = self.get_scale_weight()
            print(f"Cycle complete. Final weight: {weight} g")
            return weight

    def import_and_test_liquid_class(self, liquid_class_dict, volume):
        """
        Imports a liquid class from a dictionary, then immediately performs a 
        pipetting and weighing cycle using that class and volume.

        Args:
            liquid_class_dict (list[dict]): Dictionary defining the liquid class(es).
            volume (float): The volume to pipette in microliters.

        Returns:
            float: The final weight recorded by the scale after dispensing.
        """
        # 1. Parameter Validation (for internal use)
        try:
            # Assumes the dict is a list of one, and its name is the class being tested
            liquid_class = liquid_class_dict[0]['name']
        except (IndexError, TypeError, KeyError):
            raise ValueError("Could not extract liquid class name from dictionary.")
        
        # 2. Combined Hamilton Interface Session
        with HamiltonInterface(windowed=True, simulating=False, persistent=True) as ham_int:
            normal_logging(ham_int, os.getcwd())
            
            # --- Import Liquid Class (Step 1) ---
            print("\n--- Starting Liquid Class Import ---")
            print("Liquid class dictionary:", liquid_class_dict)
            
            # NOTE: create_liquid_class_from_dict must accept a list of dicts or be adapted
            create_liquid_class_from_dict(ham_int, liquid_class_dict) 
            print(f"Liquid class '{liquid_class}' imported successfully.")
            
            # --- Pipette and Weigh (Step 2) ---
            print(f"\n--- Starting Pipetting Cycle ---")
            print(f"Pipetting cycle: {volume}uL with liquid class '{liquid_class}'")
            
            # a. Tare Scale
            self.tare_scale() # self.tare_scale() must be adapted to use ham_int if needed

            # b. Pipette Transfer
            print(f"Aspirating and dispensing {volume}uL...")
            pip_transfer(
                ham_int,
                self.tips,
                self.source_position,
                self.lvk_vial_position,
                volumes=[volume],
                liquid_class=liquid_class,
                dispense_height=1,
            )
            print("Pipetting transfer complete.")

            # c. Get Weight
            weight = self.get_scale_weight() # self.get_scale_weight() must be adapted to use ham_int if needed
            print(f"Cycle complete. Final weight: {weight} g")
            
            return weight


    def __enter__(self):
        """Context manager entry point: connects to the robot."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point: disconnects from the robot."""
        self.disconnect()


# --- Example Usage ---
if __name__ == '__main__':
    # Configuration settings
    LAYOUT_FILE = 'LVK_deck.lay'
    SCALE_PORT = 'COM4'
    SIMULATING = True  # Set to False for a real run

    try:
        # The 'with' statement automatically handles connect() and disconnect()
        with LVKController(
            layout_file=LAYOUT_FILE,
            scale_com_port=SCALE_PORT,
            simulating=SIMULATING
        ) as controller:

            # Check initial weight after an initial tare
            controller.tare_scale()
            initial_weight = controller.get_scale_weight()
            print(f"Initial weight after tare: {initial_weight} g")

            # Define parameters for the pipetting run
            volume_to_pipette = 20  # in uL
            liquid_class_name = 'Tip_50ulFilter_Water_DispenseSurface_Empty'

            # Run the main process
            final_weight = controller.pipette_and_weigh(
                volume=volume_to_pipette,
                liquid_class=liquid_class_name
            )

            print('\n--- FINAL RESULT ---')
            print(f'Weight recorded for {volume_to_pipette}uL dispense: {final_weight} g')

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")