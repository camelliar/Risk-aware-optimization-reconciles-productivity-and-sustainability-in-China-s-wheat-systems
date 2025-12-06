import pandas as pd
import numpy as np
import rasterio
from joblib import load
import os
import gc
import traceback

# ==============================================================================
# 1. Global Configuration
# ==============================================================================

BASE_INPUT_DIR = 'D:/grid_pred/'

# Directory containing the trained model
MODEL_DIR = 'D:/grid_pred/model/irri/'  # Check this path

# Output directory
OUTPUT_DIR = 'D:/grid_pred/'  # Adjusted output path

# Mask file path
MASK_FILE_PATH = 'D:/grid_pred/irrigation_mask.tif'  # Update if needed

#Model and Feature List Names
MODEL_FILE = 'model_irri.pkl'
FEATURES_FILE = 'irri_train_features.save'

# Scenarios and Years
SCENARIOS = ['T-10%', 'T-5%', 'T0', 'T+5%', 'T+10%']
YEARS = range(2001, 2021)

# ==============================================================================
# 2. Load Resources
# ==============================================================================
try:
    print("Loading Irrigation Model...")
    # These features must include 'maxt1', 'rain1', etc. as per your training
    train_features = load(os.path.join(MODEL_DIR, FEATURES_FILE))
    model = load(os.path.join(MODEL_DIR, MODEL_FILE))
    print(f"Model loaded. Expected features: {len(train_features)}")
except Exception as e:
    print(f"Failed to load model or features: {e}")
    exit()


# ==============================================================================
# 3. Helper Function (Masking & Saving)
# ==============================================================================
def save_masked_raster(data, meta, output_path, mask_path):
    try:
        with rasterio.open(mask_path) as mask_src:
            mask_data = mask_src.read(1)
            if mask_src.nodata is not None:
                valid_mask = (mask_data != mask_src.nodata)
            else:
                valid_mask = ~np.isnan(mask_data)

        masked_data = np.where(valid_mask, data, np.nan)

        meta.update({
            'driver': 'GTiff', 'dtype': 'float32', 'nodata': np.nan,
            'count': 1, 'compress': 'lzw'
        })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(masked_data.astype(np.float32), 1)
        print(f" Saved: {output_path}")

    except Exception as e:
        print(f" Save Warning: {e}")


# ==============================================================================
# 4. Main Prediction Loop
# ==============================================================================

for scenario in SCENARIOS:
    print(f"\n Processing Scenario: {scenario}")

    for year in YEARS:
        print(f" Year: {year} ...", end=" ")

        clim_dir = os.path.join(BASE_INPUT_DIR, 'climate')
        soil_dir = os.path.join(BASE_INPUT_DIR, 'soil')
        latlon_dir = os.path.join(BASE_INPUT_DIR, 'latlon')

        # ---------------------------------------------------------
        # A. Feature Mapping (Matched to your OLD code)
        # ---------------------------------------------------------
        feature_files = {
            # Standard Climate
            'rain': os.path.join(clim_dir, f'pre{year}.tif'),
            'tmax': os.path.join(clim_dir, f'maxt{year}.tif'),
            'srad': os.path.join(clim_dir, f'rad{year}.tif'),

            # Geography
            'lon': os.path.join(latlon_dir, 'lon1.tif'),
            'lat': os.path.join(latlon_dir, 'lat1.tif'),

            # Soil Properties
            'AWC': os.path.join(soil_dir, 'AWC.tif'),
            'SBDM_mean': os.path.join(soil_dir, 'BULK.tif'),
            'SLOC_mean': os.path.join(soil_dir, 'org_carbon.tif'),
            'SLCL_mean': os.path.join(soil_dir, 'clay.tif'),
            'SLSI_mean': os.path.join(soil_dir, 'silt.tif'),
            'SLHW_mean': os.path.join(soil_dir, 'ph_water.tif'),

            # Zone
            'zone': os.path.join(BASE_INPUT_DIR, 'zone/zone.tif'),

            # --- Specific Seasonal Features---
            'maxt1': os.path.join(clim_dir, f'maxt_p1_{year}.tif'),
            'rain1': os.path.join(clim_dir, f'pre_p1_{year}.tif'),
            'srad1': os.path.join(clim_dir, f'rad_p1_{year}.tif'),
            'maxt2': os.path.join(clim_dir, f'maxt_p2_{year}.tif'),
            'rain2': os.path.join(clim_dir, f'pre_p2_{year}.tif'),
            'srad2': os.path.join(clim_dir, f'rad_p2_{year}.tif'),
        }

        try:
            # --- B. Read Rasters ---
            data_list = []
            col_names = []
            meta = None
            height, width = 0, 0

            with rasterio.open(feature_files['lon']) as src:
                meta = src.meta.copy()
                height, width = src.height, src.width

            for name, path in feature_files.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")

                with rasterio.open(path) as src:
                    arr = src.read(1).flatten().astype(np.float32)
                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan
                    data_list.append(arr)
                    col_names.append(name)

            df_grid = pd.DataFrame(np.column_stack(data_list), columns=col_names, dtype=np.float32)
            del data_list
            gc.collect()

            # --- C. Preprocessing ---
            df_grid['zone'] = df_grid['zone'].fillna(0).astype(int).astype(str)
            df_grid['scenario'] = scenario

            # --- D. Encoding & Alignment ---
            # Encode only zone and scenario
            df_grid_encoded = pd.get_dummies(df_grid, columns=['zone', 'scenario'])

            df_final = df_grid_encoded.reindex(columns=train_features, fill_value=0)

            # --- E. Prediction ---
            pred_log = model.predict(df_final)

            pred_values = np.expm1(pred_log)
            pred_values = np.maximum(pred_values, 0)  # Clip negative values

            # --- F. Saving ---
            pred_raster = pred_values.reshape(height, width)
            output_file = os.path.join(OUTPUT_DIR, scenario, f'irri{year}.tif')
            save_masked_raster(pred_raster, meta, output_file, MASK_FILE_PATH)

            # --- G. Cleanup ---
            del df_grid, df_grid_encoded, df_final, pred_log, pred_values, pred_raster
            gc.collect()

        except Exception as e:
            print(f"\n Error in {scenario} - {year}: {e}")
            traceback.print_exc()
