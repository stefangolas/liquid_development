# Assuming these imports are necessary from your project's environment
from pyhamilton import (HamiltonInterface, LayoutManager,
                        layout_item, ReagentTrackedBulkPlate,
                        Tip96, TrackedTips, LVKBalanceVial,
                        create_liquid_class_from_json)
from pyhamilton.pipetting import pip_transfer
from mettler_toledo import MettlerWXS

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

    def import_liquid_classes_from_file(self, json_file_path):
        """
        Applies liquid classes from a specified JSON file.

        Args:
            json_file_path (str): The path to the liquid class JSON file.
        """

        print(f"Importing liquid classes from {json_file_path}...")
        with HamiltonInterface(windowed=True, simulating=self.simulating, persistent=True) as ham_int:
            create_liquid_class_from_json(ham_int, json_file_path)
        print("Liquid classes imported successfully.")

    def tare_scale(self):
        """Tares the Mettler scale to zero and confirms."""
        print("Taring scale...")
        # USE THE CONTEXT MANAGER
        with MettlerWXS(self.scale_com_port, simulating=self.simulating) as scale:
            scale.tare(immediately=True)
        print("Scale tared.")

    def get_scale_weight(self):
        """
        Retrieves the current weight from the Mettler scale.
        ...
        """
        # USE THE CONTEXT MANAGER
        with MettlerWXS(self.scale_com_port, simulating=self.simulating) as scale:
            weight = scale.get_weight(immediately=True)
            print(f"Weight measured: {weight} g")
            return weight

    def pipette_and_weigh(self, volume, liquid_class):
        """
        Performs a full pipetting and weighing cycle.

        Args:
            volume (float): The volume to pipette in microliters.
            liquid_class (str): The name of the liquid class to use for the transfer.

        Returns:
            float: The final weight recorded by the scale after dispensing.
        """
        with HamiltonInterface(windowed=True, simulating=self.simulating, persistent=True) as ham_int:
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