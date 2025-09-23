from pyhamilton import HamiltonInterface, layout_item, LayoutManager
from pyhamilton.pipetting import pip_transfer, multi_dispense
from pyhamilton.resources import ResourceType, LVKBalanceVial
from pyhamilton.consumables import ReagentTrackedBulkPlate

from mettler_toledo import MettlerWXS

lmgr = LayoutManager('deck.lay')
lvk_vial = layout_item(lmgr, LVKBalanceVial, 'LVK_BALANCE_VIAL_0001')
lvk_vial_position = [(lvk_vial, 0)]

mettler_scale = MettlerWXS('COM3', simulating=True)

with HamiltonInterface() as ham_int:
    weight = mettler_scale.get_weight()