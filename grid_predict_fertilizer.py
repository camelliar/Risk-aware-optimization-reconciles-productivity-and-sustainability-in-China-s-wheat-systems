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
# Base directory for input raster data
BASE_INPUT_DIR = 'D:/grid_pred/'
MODEL_DIR = 'D:/grid_pred/model/'
OUTPUT_DIR = 'D:/grid_pred/'
MASK_FILE_PATH = 'D:/grid_pred/fertilizer_mask.tif'
MODEL_FILE = 'model_fer.pkl'
FEATURES_FILE = 'fer_train_features.save'
# Scenarios and Years to process
SCENARIOS = ['T-10%', 'T-5%', 'T0', 'T+5%', 'T+10%']
YEARS = range(2001, 2021)
# ==============================================================================
# 2. Load Resources
# ==============================================================================
try:
    train_features = load(os.path.join(MODEL_DIR, FEATURES_FILE))
    model = load(os.path.join(MODEL_DIR, MODEL_FILE))
    print(f"Model loaded successfully. Expected features: {len(train_features)}")
except Exception as e:
    print(f"Failed to load model or features: {e}")
    exit()
# ==============================================================================
# 3. Helper Function
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
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': np.nan,
            'count': 1,
            'compress': 'lzw'  # Compression to reduce file size
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
        print(f"Year: {year} ...", end=" ")

        # --- A. Define File Paths ---
        clim_dir = os.path.join(BASE_INPUT_DIR, 'climate')
        soil_dir = os.path.join(BASE_INPUT_DIR, 'soil')
        latlon_dir = os.path.join(BASE_INPUT_DIR, 'latlon')

        feature_files = {
            'rain': os.path.join(clim_dir, f'pre{year}.tif'),
            'tmax': os.path.join(clim_dir, f'maxt{year}.tif'),
            'tmin': os.path.join(clim_dir, f'mint{year}.tif'),
            'srad': os.path.join(clim_dir, f'rad{year}.tif'),
            'rain_more_50': os.path.join(clim_dir, f'epre{year}.tif'),
            'maxt_sum2': os.path.join(clim_dir, f'etmp{year}.tif'),

            'lon': os.path.join(latlon_dir, 'lon1.tif'),
            'lat': os.path.join(latlon_dir, 'lat1.tif'),

            'AWC': os.path.join(soil_dir, 'AWC.tif'),
            'SBDM_mean': os.path.join(soil_dir, 'BULK.tif'),
            'SLOC_mean': os.path.join(soil_dir, 'org_carbon.tif'),
            'SLCL_mean': os.path.join(soil_dir, 'clay.tif'),
            'SLSI_mean': os.path.join(soil_dir, 'silt.tif'),
            'SLHW_mean': os.path.join(soil_dir, 'ph_water.tif'),
            'SCEC_mean': os.path.join(soil_dir, 'cec_clay.tif'),

            'zone': os.path.join(BASE_INPUT_DIR, 'zone/zone.tif'),
            'irrigation': os.path.join(BASE_INPUT_DIR, 'irrigation/irrigation.tif')
        }

        try:
            # --- B. Read Rasters & Build Matrix ---
            data_list = []
            col_names = []
            meta = None
            height, width = 0, 0

            # Get metadata from the first file
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

            # Free up memory immediately
            del data_list
            gc.collect()

            # --- C. Preprocessing ---
            df_grid['zone'] = df_grid['zone'].fillna(0).astype(int).astype(str)

            df_grid['irrigation'] = df_grid['irrigation'].fillna(0)
            irrigation_map = {1.0: 'irrigated', 0.0: 'rainfed'}
            df_grid['irrigation'] = df_grid['irrigation'].map(irrigation_map).fillna('rainfed')

            # Scenario: Assign current scenario
            df_grid['scenario'] = scenario

            # --- D. Encoding & Alignment (CRITICAL) ---
            cols_to_encode = ['zone', 'irrigation', 'scenario']
            df_grid_encoded = pd.get_dummies(df_grid, columns=cols_to_encode)

            # Reindex to strictly match training features
            df_final = df_grid_encoded.reindex(columns=train_features, fill_value=0)

            # --- E. Prediction & Saving ---
            pred_values = model.predict(df_final)

            # Reshape back to 2D raster dimensions
            pred_raster = pred_values.reshape(height, width)

            # Construct output path and save
            output_file = os.path.join(OUTPUT_DIR, scenario, f'fer{year}.tif')
            save_masked_raster(pred_raster, meta, output_file, MASK_FILE_PATH)

            # --- F. Cleanup ---
            del df_grid, df_grid_encoded, df_final, pred_values, pred_raster
            gc.collect()

        except Exception as e:
            print(f"\n Error in {scenario} - {year}: {e}")
            traceback.print_exc()