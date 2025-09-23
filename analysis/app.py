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
from pyhamilton.liquid_class_db import get_liquid_class_parameter, create_correction_curve, unpack_doubles_dynamic

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

def load_models():
    """
    Loads the trained models from the joblib file and sets up the feature list.
    This function is called once when the application starts.
    """
    global MODELS, ALL_FEATURES, MODEL_FEATURES
    model_path = 'xgb_models.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run the training script first.")
        exit(1)

    print("Loading XGBoost models...")
    MODELS = joblib.load(model_path)
    
    all_unique_features = set()
    for target_col, model in MODELS.items():
        features = model.get_booster().feature_names
        MODEL_FEATURES[target_col] = features
        all_unique_features.update(features)
    
    ALL_FEATURES = sorted(list(all_unique_features))
    
    print(f"Models loaded successfully for targets: {list(MODELS.keys())}")
    print(f"Total features to display: {len(ALL_FEATURES)}")

# Initialize on startup
load_models()
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
    Get the full details of a specific liquid class by name.
    """
    if not LIQUID_CLASSES_CACHE:
        return jsonify({'error': 'No liquid classes loaded'}), 500
    
    for lc in LIQUID_CLASSES_CACHE:
        if lc['LiquidClassName'] == class_name:
            # We already unpacked the correction curve during loading, so just return the dict
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


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives user input via a JSON payload, performs predictions on missing fields,
    and returns the predicted values.
    """
    input_data = request.json
    
    # Convert input data to a Pandas Series
    input_series = pd.Series(input_data)
    
    predictions = {}
    
    # Find which features are missing from the input
    missing_features = [f for f in ALL_FEATURES if f not in input_series or pd.isna(input_series.get(f))]
    
    for target_col in missing_features:
        if target_col in MODELS:
            model = MODELS[target_col]
            features_for_this_model = MODEL_FEATURES.get(target_col, [])
            
            try:
                # FIX: Removed the incorrect 'dtype' argument from the reindex call
                input_df = input_series.reindex(features_for_this_model).to_frame().T
                
                prediction = model.predict(input_df)[0]
                
                # For DispenseMode, which is a categorical variable, we should cast to an integer
                if target_col == 'DispenseMode':
                    predictions[target_col] = int(round(prediction))
                else:
                    predictions[target_col] = float(prediction)
            
            except Exception as e:
                print(f"Error predicting for {target_col}: {e}")
                predictions[target_col] = None

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)