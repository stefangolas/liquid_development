from pyhamilton import HamiltonInterface, layout_item, LayoutManager, Plate96
from pyhamilton.pipetting import pip_transfer, multi_dispense
from pyhamilton.resources import ResourceType, LVKBalanceVial
from pyhamilton.consumables import ReagentTrackedBulkPlate
from pyhamilton.liquid_classes import create_liquid_class_from_json

from mettler_toledo import MettlerWXS

lmgr = LayoutManager('LVK_deck.lay')
lvk_vial = layout_item(lmgr, LVKBalanceVial, 'LVK_BALANCE_VIAL_0001')
lvk_vial_position = [(lvk_vial, 0)]

source_plate = layout_item(lmgr, Plate96, 'Test_HSP')
source_position = [(source_plate, 0)]

mettler_scale = MettlerWXS('COM3', simulating=True)

def import_liquid_class(ham, liquid_class_name):
    """
    Applies the specified liquid class to the Hamilton interface.
    """
    with HamiltonInterface(windowed=True) as ham_int:
        ham_int.initialize()
        create_liquid_class_from_json(ham_int, 'my_lcs.json')

def pipette_and_record(volume, liquid_class):
    with HamiltonInterface(windowed=True, persistent=True) as ham_int:
        mettler_scale.tare(immediately=True)
        pip_transfer(ham_int, volume, source_position, lvk_vial_position, liquid_class=liquid_class)
        weight = mettler_scale.get_weight(immediately=True)
        return weight

def initialize():
    with HamiltonInterface(windowed=True, persistent=True) as ham_int:
        ham_int.initialize()