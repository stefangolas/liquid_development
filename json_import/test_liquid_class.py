from pyhamilton import HamiltonInterface, copy_liquid_class, create_liquid_class_from_json

with HamiltonInterface(windowed=True) as ham_int:
    ham_int.initialize()
    create_liquid_class_from_json(ham_int, 'my_lcs.json')