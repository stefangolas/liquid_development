import os
import json
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import text
from urllib.parse import quote_plus
from importlib import util
from pyhamilton.defaults import defaults
from pyhamilton.liquid_class_db import get_liquid_class_parameter, create_correction_curve, unpack_doubles_dynamic, DispenseMode
from pyhamilton.ngs import get_last_usb_data_block

from lvk_script import LVKController

LVK_ROBOT = LVKController('LVK_Deck.lay', 'COM4', simulating=False)

# A dictionary to hold our trained XGBoost models. We will load them once at startup.
MODELS = None
# A list of all the feature names used across all models for building the form.
ALL_FEATURES = None
# A dictionary to map each target column to its specific feature list.
MODEL_FEATURES = {}
# A dictionary to map machine-readable names to human-readable names.
FEATURE_LABELS = {
    'AsFlowRate': 'Aspirate Flow Rate',
    'AsMixFlowRate': 'Aspirate Mix Flow Rate',
    'AsAirTransportVolume': 'Aspirate Air Transport Volume',
    'AsBlowOutVolume': 'Aspirate Blow Out Volume',
    'AsSwapSpeed': 'Aspirate Swap Speed',
    'AsSettlingTime': 'Aspirate Settling Time',
    'AsOverAspirateVolume': 'Aspirate Overaspirate Volume',
    'AsClotRetractHeight': 'Aspirate Clot Retract Height',
    'DsFlowRate': 'Dispense Flow Rate',
    'DsMixFlowRate': 'Dispense Mix Flow Rate',
    'DsAirTransportVolume': 'Dispense Air Transport Volume',
    'DsBlowOutVolume': 'Dispense Blow Out Volume',
    'DsSwapSpeed': 'Dispense Swap Speed',
    'DsSettlingTime': 'Dispense Settling Time',
    'DsStopFlowRate': 'Dispense Stop Flow Rate',
    'DsStopBackVolume': 'Dispense Stop Back Volume',
    'DispenseMode': 'Dispense Mode',
    'TipType': 'Tip Type'
}

# Group features for a cleaner frontend layout.
ASPIRATE_FEATURES = [
    'AsFlowRate', 'AsMixFlowRate', 'AsAirTransportVolume',
    'AsBlowOutVolume', 'AsSwapSpeed', 'AsSettlingTime',
    'AsOverAspirateVolume', 'AsClotRetractHeight'
]
DISPENSE_FEATURES = [
    'DsFlowRate', 'DsMixFlowRate', 'DsAirTransportVolume',
    'DsBlowOutVolume', 'DsSwapSpeed', 'DsSettlingTime',
    'DsStopFlowRate', 'DsStopBackVolume', 'DispenseMode', 'TipType'
]

# A simple Flask application instance
app = Flask(__name__)

# Cache for liquid classes to avoid repeated database queries
LIQUID_CLASSES_CACHE = None

def _check_access_dialect():
    """Raise if `sqlalchemy-access` is not installed."""
    if util.find_spec("sqlalchemy_access") is None:
        raise ModuleNotFoundError(
            "SQLAlchemy Access dialect not found. "
            "Install with: pip install sqlalchemy-access"
        )

def _build_engine(mdb_path):
    """Build SQLAlchemy engine for Access database."""
    _check_access_dialect()
    
    driver = "Microsoft Access Driver (*.mdb, *.accdb)"
    odbc_str = f"DRIVER={{{driver}}};DBQ={mdb_path};"
    uri = f"access+pyodbc:///?odbc_connect={quote_plus(odbc_str)}"
    
    from sqlalchemy import create_engine
    return create_engine(uri, future=True)

def load_liquid_classes():
    """
    Load liquid classes from the Access database into memory for fast searching.
    This is called once at startup and cached.
    """
    global LIQUID_CLASSES_CACHE
    
    try:
        # Get the database path from your config
        cfg = defaults()
        engine = _build_engine(cfg.liquids_database)
        
        param_columns = [
            'LiquidClassName',
            'AsFlowRate', 'AsMixFlowRate', 'AsAirTransportVolume', 'AsBlowOutVolume', 
            'AsSwapSpeed', 'AsSettlingTime', 'AsOverAspirateVolume', 'AsClotRetractHeight', 
            'DsFlowRate', 'DsMixFlowRate', 'DsAirTransportVolume', 'DsBlowOutVolume', 
            'DsSwapSpeed', 'DsSettlingTime', 'DsStopFlowRate', 'DsStopBackVolume', 
            'DispenseMode', 'TipType', 'CorrectionCurve'
        ]
        
        select_string = ", ".join(param_columns)
        query = f"SELECT {select_string} FROM LiquidClass WHERE OriginalLiquid = 0"
        stmt = text(query)
        
        with engine.connect() as conn:
            result = conn.execute(stmt).fetchall()
        
        LIQUID_CLASSES_CACHE = []
        for row in result:
            lc_data = dict(row._mapping)
            # Unpack the CorrectionCurve for the API response
            if 'CorrectionCurve' in lc_data and lc_data['CorrectionCurve']:
                try:
                    unpacked_data = unpack_doubles_dynamic(lc_data['CorrectionCurve'])
                    lc_data['CorrectionCurve'] = unpacked_data
                except Exception as e:
                    print(f"Failed to unpack CorrectionCurve for {lc_data['LiquidClassName']}: {e}")
                    lc_data['CorrectionCurve'] = None
            
            LIQUID_CLASSES_CACHE.append(lc_data)
        
        print(f"Loaded {len(LIQUID_CLASSES_CACHE)} liquid classes into cache")
        # Debug: Print first few liquid class names
        if LIQUID_CLASSES_CACHE:
            print("Sample liquid classes:")
            for i, lc in enumerate(LIQUID_CLASSES_CACHE[:5]):
                print(f"  - {lc.get('LiquidClassName', 'Unknown')}")
        
    except Exception as e:
        print(f"Warning: Could not load liquid classes from database: {e}")
        LIQUID_CLASSES_CACHE = []

load_liquid_classes()

@app.route('/')
def home():
    """
    Renders the main page of the application, passing organized feature lists
    and their labels to the template.
    """
    return render_template(
        'index.html',
        aspirate_features=ASPIRATE_FEATURES,
        dispense_features=DISPENSE_FEATURES,
        feature_labels=FEATURE_LABELS
    )

@app.route('/search_liquid_classes', methods=['GET'])
def search_liquid_classes():
    """
    Search for liquid classes by name. Returns matches for autocomplete.
    """
    search_term = request.args.get('q', '').lower()
    
    print(f"Search request received for: '{search_term}'")  # Debug log
    
    if not search_term:
        print("Empty search term")
        return jsonify([])
    
    if not LIQUID_CLASSES_CACHE:
        print("Liquid classes cache is empty")
        return jsonify({'error': 'No liquid classes loaded'}), 500
    
    # Find matching liquid classes (case-insensitive partial match)
    matches = []
    for lc in LIQUID_CLASSES_CACHE:
        lc_name = lc.get('LiquidClassName', '')
        if lc_name and search_term in lc_name.lower():
            # Return only name for autocomplete
            matches.append({
                'name': lc_name
            })
    
    print(f"Found {len(matches)} matches for '{search_term}'")  # Debug log
    
    # Limit results for performance
    matches = matches[:20]
    
    return jsonify(matches)

@app.route('/get_liquid_class/<class_name>', methods=['GET'])
def get_liquid_class(class_name):
    """
    Get the full details of a specific liquid class by name, converting
    dispense mode code to a human-readable string.
    """
    print(f"Get request for liquid class: '{class_name}'")  #
    if not LIQUID_CLASSES_CACHE:
        return jsonify({'error': 'No liquid classes loaded'}), 500
    print(f"Fetching details for liquid class: '{class_name}'")  # Debug log
    for lc in LIQUID_CLASSES_CACHE:
        if lc['LiquidClassName'] == class_name:
            # Convert dispense mode code to string
            if 'DispenseMode' in lc:
                code = lc['DispenseMode']
                try:
                    lc['DispenseMode'] = DispenseMode.from_code(code).value
                except ValueError:
                    lc['DispenseMode'] = code
            
            # Return the dict as JSON
            print(f"Returning liquid class: {lc}")  # Debug log
            return jsonify(lc)

    return jsonify({'error': 'Liquid class not found'}), 404

@app.route('/test_liquid_classes', methods=['GET'])
def test_liquid_classes():
    """
    Test endpoint to verify liquid classes are loaded.
    """
    if not LIQUID_CLASSES_CACHE:
        return jsonify({'status': 'error', 'message': 'No liquid classes loaded', 'count': 0})
    
    # Return first 5 liquid class names as a test
    sample_names = [lc.get('LiquidClassName', 'Unknown') for lc in LIQUID_CLASSES_CACHE[:5]]
    
    return jsonify({
        'status': 'success',
        'count': len(LIQUID_CLASSES_CACHE),
        'sample_names': sample_names
    })

LIQUID_CLASSES_FILE = 'liquid_classes.json'


@app.route('/save_liquid_class', methods=['POST'])
def save_liquid_class():
    """
    Saves a new or updated liquid class to the liquid_classes.json file.
    """
    try:
        new_liquid_class = request.json
        if not new_liquid_class or "name" not in new_liquid_class:
            return jsonify({'error': 'Invalid data: "name" field is missing.'}), 400

        # Load existing liquid classes
        liquid_classes = []
        if os.path.exists(LIQUID_CLASSES_FILE):
            with open(LIQUID_CLASSES_FILE, 'r') as f:
                try:
                    liquid_classes = json.load(f)
                except json.JSONDecodeError:
                    pass  # File is empty or invalid, start with an empty list

        # Check for and replace existing liquid class with the same name
        found = False
        for i, lc in enumerate(liquid_classes):
            if lc.get('name') == new_liquid_class.get('name'):
                liquid_classes[i] = new_liquid_class
                found = True
                break
        if not found:
            liquid_classes.append(new_liquid_class)

        # Save the updated list back to the file with indentation for readability
        with open(LIQUID_CLASSES_FILE, 'w') as f:
            json.dump(liquid_classes, f, indent=4)
        
        return jsonify({'status': 'success', 'message': f'Liquid class "{new_liquid_class["name"]}" saved successfully.'}), 200

    except Exception as e:
        print(f"Error saving liquid class: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/initialize_hamilton', methods=['POST'])
def initialize_hamilton():
    """
    Initializes the connection to the Hamilton robot using the LVKController.
    """
    # Assuming LVK_ROBOT is a global instance of LVKController
    if LVK_ROBOT is None:
        return jsonify({'status': 'error', 'message': 'LVKController not initialized'}), 500
        
    try:
        LVK_ROBOT.connect()
        return jsonify({'status': 'success', 'message': 'Hamilton initialized successfully.'}), 200
    except ConnectionError as e:
        # Catch the specific error raised by LVKController.connect() for a clean message
        return jsonify({'status': 'error', 'message': f'Connection Error: {e}'}), 500
    except Exception as e:
        print(f"Error during Hamilton initialization: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/import_liquid_class', methods=['POST'])
def import_liquid_class():
    """
    Imports liquid classes from a specified JSON file into the Hamilton robot.
    """
    # Assuming LVK_ROBOT is a global instance of LVKController
    if LVK_ROBOT is None:
        return jsonify({'status': 'error', 'message': 'LVKController not initialized'}), 500

    data = request.json
    liquid_class_dict = data.get('liquid_class_dictionary')

    if not liquid_class_dict:
        return jsonify({'error': 'Invalid or missing liquid_class_dictionary'}), 400

    try:
        print("Importing liquid class dictionary:", liquid_class_dict)
        LVK_ROBOT.import_liquid_class_from_dictionary(liquid_class_dict)
        return jsonify({'status': 'success', 'message': f'Liquid class imported successfully.'}), 200
    except ConnectionError as e:
        # Indicate if the robot wasn't connected/initialized
        return jsonify({'status': 'error', 'message': f'Connection Error: {e}. Is the Hamilton initialized?'}), 500

@app.route('/import_and_test_liquid_class', methods=['POST'])
def import_and_test_liquid_class_endpoint():
    """
    Imports liquid classes and immediately performs a pipetting and weighing test.
    """
    # 1. Initialization Check
    if LVK_ROBOT is None:
        return jsonify({'status': 'error', 'message': 'LVKController not initialized'}), 500

    data = request.json
    liquid_class_dict = data.get('liquid_class_dictionary')
    volume = data.get('volume')

    # 2. Parameter Validation
    if not liquid_class_dict or not isinstance(liquid_class_dict, list):
        return jsonify({'error': 'Missing or invalid liquid_class_dictionary (must be a list)'}), 400
    if volume is None:
        return jsonify({'error': 'Missing required parameter: volume'}), 400
    
    try:
        volume_float = float(volume)
        liquid_class_name = liquid_class_dict[0]['name']
    except (TypeError, ValueError, IndexError, KeyError) as e:
        return jsonify({'error': f'Invalid volume format or missing liquid class name in dictionary: {e}'}), 400

    # 3. Call the Combined LVKController Method
    try:
        weight = LVK_ROBOT.import_and_test_liquid_class(liquid_class_dict, volume_float)
        
        # Assuming this function call is still necessary/valid after the robot work
        tadm_data = get_last_usb_data_block()['channels'] 

        if weight is None:
            return jsonify({'error': 'Failed to get weight from scale (Weight is None)'}), 500

        # 4. Success Response
        return jsonify({
            'status': 'success',
            'message': f'Liquid class "{liquid_class_name}" imported and test complete.',
            'test_result': {
                'weight': weight, 
                'volume': volume_float, 
                'liquid_class': liquid_class_name,
                'tadm_data': tadm_data
            }
        }), 200

    except ConnectionError as e:
        return jsonify({'status': 'error', 'message': f'Connection Error: {e}. Is the Hamilton initialized?'}), 500
    except ValueError as e:
        # Handle the ValueError raised from within the LVKController method
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({'status': 'error', 'message': f'An unexpected error occurred during test: {e}'}), 500


if __name__ == '__main__':
    app.run(debug=True)