import pandas as pd
import numpy as np
import os
from src.knn import Knn, mean_absolute_error, r_squared
from src.DataImportMethods import runAllMetaImports, getMetaDF, load_image_data
from src.feature_extraction import extract_hog, extract_hog_features
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import shutil
import tempfile
import time
import math
import gc  # Garbage collector

# === File Paths ===
meta_path = "./wiki_crop/wiki.mat"
image_dir = "./wiki_crop"
meta_csv_path = "./wiki_crop/meta.csv"
struct_key = "wiki"
xTrain_path = "./wiki_crop/xTrain.csv"
yTrain_path = "./wiki_crop/yTrain.csv"
xTest_path = "./wiki_crop/xTest.csv"
yTest_path = "./wiki_crop/yTest.csv"

# === Step 1: Convert .mat to .csv ===
runAllMetaImports(image_dir, meta_path, struct_key, meta_csv_path)
df = getMetaDF(meta_csv_path)

# Debug step - show the first few paths to understand structure
print("First few paths from metadata:")
print(df['full_path'].head(5))

# === Choose feature extraction method ===
USE_HOG = True  # Set to False to use raw pixels instead

if USE_HOG:
    print("Extracting HOG features...")
    # Fix paths for consistency
    df['normalized_path'] = df['full_path'].apply(
        lambda x: os.path.normpath(
            x.replace('wiki_crop/', '') if x.startswith('wiki_crop/') else x
        )
    )
    
    # Create full paths for image loading
    full_paths = [os.path.join(image_dir, path.replace('\\', '/')) 
                 for path in df['normalized_path']]
    
    # Debug - check if files exist
    for i, path in enumerate(full_paths[:10]):  # Check first 10 paths
        print(f"Path {i}: {path}, exists: {os.path.exists(path)}")
    
    # Extract features in parallel
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(extract_hog)(path) for path in full_paths
    )
    
    # Filter out invalid results
    valid = [(i, feat) for i, feat in enumerate(results) if feat is not None]
    if not valid:
        raise ValueError("No valid images were loaded with HOG. Check paths.")
        
    indices, features = zip(*valid)
    filtered_df = df.iloc[list(indices)]
    
    # Calculate age
    y_data = filtered_df['photo_taken'] - (filtered_df['dob'] / 365.25 + 1969)
    
    # Get feature dimension
    feature_dim = features[0].shape[0]
    print(f"Feature dimension: {feature_dim}")
    print(f"Successfully extracted features from {len(features)} images")
    
    # Handle memory constraints - process in batches for train/test split
    print("Splitting data...")
    # First split indices to avoid memory issues
    indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Process training data in batches
    print("Processing training data...")
    batch_size = 5000  # Adjust based on available memory
    total_batches = len(train_indices) // batch_size + (1 if len(train_indices) % batch_size > 0 else 0)
    
    # Create empty dataframes for train data
    x_train_df = pd.DataFrame()
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(train_indices))
        print(f"Processing training batch {i+1}/{total_batches}")
        
        # Get batch indices
        batch_indices = train_indices[start_idx:end_idx]
        
        # Extract features for this batch
        batch_features = [features[idx] for idx in batch_indices]
        
        # Convert to DataFrame and append
        batch_df = pd.DataFrame(batch_features)
        x_train_df = pd.concat([x_train_df, batch_df], ignore_index=True)
        
        # Clean up to free memory
        del batch_features, batch_df
        gc.collect()
    
    # Save training features
    print("Saving training features...")
    x_train_df.to_csv(xTrain_path, index=False)
    
    # Save training labels
    y_train = y_data[train_indices]
    pd.DataFrame(y_train, columns=['age']).to_csv(yTrain_path, index=False)
    
    # Process test data
    print("Processing test data...")
    x_test_df = pd.DataFrame()
    total_test_batches = len(test_indices) // batch_size + (1 if len(test_indices) % batch_size > 0 else 0)
    
    for i in range(total_test_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(test_indices))
        print(f"Processing test batch {i+1}/{total_test_batches}")
        
        # Get batch indices
        batch_indices = test_indices[start_idx:end_idx]
        
        # Extract features for this batch
        batch_features = [features[idx] for idx in batch_indices]
        
        # Convert to DataFrame and append
        batch_df = pd.DataFrame(batch_features)
        x_test_df = pd.concat([x_test_df, batch_df], ignore_index=True)
        
        # Clean up to free memory
        del batch_features, batch_df
        gc.collect()
    
    # Save test features
    print("Saving test features...")
    x_test_df.to_csv(xTest_path, index=False)
    
    # Save test labels
    y_test = y_data[test_indices]
    pd.DataFrame(y_test, columns=['age']).to_csv(yTest_path, index=False)
    
    # Clean up to free memory
    del features, x_train_df, x_test_df
    gc.collect()
    
else:
    # Use the standard method with pixel features
    print("Loading images with pixel features...")
    x_data, y_data, filtered_df = load_image_data(df, image_dir)
    
    if len(x_data) == 0:
        raise ValueError("No valid images were loaded. Check paths.")
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42)
        
    # Save data
    pd.DataFrame(x_train).to_csv(xTrain_path, index=False)
    pd.DataFrame(x_test).to_csv(xTest_path, index=False)
    pd.DataFrame(y_train, columns=['age']).to_csv(yTrain_path, index=False)
    pd.DataFrame(y_test, columns=['age']).to_csv(yTest_path, index=False)

# === Step 4: Train and evaluate KNN but also in batches to handle memory constraints ===
print("Loading training data...")
# Load the data in chunks
chunk_size = 5000  # Adjust based on available memory
x_train = pd.read_csv(xTrain_path, chunksize=chunk_size)
y_train = pd.read_csv(yTrain_path)['age'].values

# Combine the chunks
x_train_data = []
for chunk in x_train:
    x_train_data.append(chunk.values)
x_train = np.vstack(x_train_data)

print("Training KNN model...")
knn = Knn(k=5)
knn.train(x_train, y_train)

print("Loading test data...")
# Load test data in chunks
x_test = pd.read_csv(xTest_path, chunksize=chunk_size)
y_test = pd.read_csv(yTest_path)['age'].values

# Make predictions in batches
print("Making predictions...")
all_predictions = []

for chunk in x_test:
    batch_predictions = knn.predict(chunk.values)
    all_predictions.append(batch_predictions)

yHat = np.concatenate(all_predictions)

print("Evaluation Metrics:")
print("-------------------")
print(f"MAE: {mean_absolute_error(yHat, y_test):.2f} years")
print(f"R^2: {r_squared(yHat, y_test):.3f}")

def convert_size(size_bytes):
    """Convert bytes to a human-readable format"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.log(size_bytes, 1024))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def get_size(path):
    """Get size of a directory"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size

def clean_temp_files():
    """Clean temporary files and report space freed"""
    # Define paths to clean
    temp_dirs = [
        tempfile.gettempdir(),  # System temp directory
        os.path.join(os.environ.get('USERPROFILE', ''), 'AppData', 'Local', 'Temp'),  # Windows temp
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Temp'),  # Another Windows temp location
        os.path.join(os.path.expanduser('~'), '.cache')  # Cache directory
    ]
    
    # Add Python-specific cache directories
    python_cache_dirs = [
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache'
    ]
    
    # Get current directory
    current_dir = os.getcwd()
    
    print(f"Current directory: {current_dir}")
    print("Cleaning temporary files...")
    
    total_freed = 0
    
    # Clean system temp directories
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            before_size = get_size(temp_dir)
            print(f"\nScanning: {temp_dir}")
            print(f"Size before cleaning: {convert_size(before_size)}")
            
            try:
                # List all entries in the temp directory
                entries = os.listdir(temp_dir)
                deleted = 0
                for entry in entries:
                    try:
                        full_path = os.path.join(temp_dir, entry)
                        # Skip if the file is in use
                        if os.path.isfile(full_path):
                            try:
                                os.remove(full_path)
                                deleted += 1
                            except (PermissionError, OSError):
                                pass
                        elif os.path.isdir(full_path):
                            try:
                                shutil.rmtree(full_path)
                                deleted += 1
                            except (PermissionError, OSError):
                                pass
                    except Exception as e:
                        pass
                
                after_size = get_size(temp_dir)
                freed = before_size - after_size
                total_freed += freed
                print(f"Deleted {deleted} items")
                print(f"Size after cleaning: {convert_size(after_size)}")
                print(f"Space freed: {convert_size(freed)}")
            except Exception as e:
                print(f"Error cleaning {temp_dir}: {e}")
    
    # Find and clean Python cache directories in the project
    for root, dirs, files in os.walk(current_dir):
        for cache_dir in python_cache_dirs:
            if cache_dir in dirs:
                cache_path = os.path.join(root, cache_dir)
                try:
                    before_size = get_size(cache_path)
                    print(f"\nCleaning Python cache: {cache_path}")
                    print(f"Size before cleaning: {convert_size(before_size)}")
                    shutil.rmtree(cache_path)
                    total_freed += before_size
                    print(f"Deleted cache directory")
                    print(f"Space freed: {convert_size(before_size)}")
                except Exception as e:
                    print(f"Error cleaning {cache_path}: {e}")
    
    print(f"\nTotal space freed: {convert_size(total_freed)}")
    
    # Check for NumPy memory-mapped files
    numpy_files = []
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.npy') or file.endswith('.npz'):
                numpy_files.append(os.path.join(root, file))
    
    if numpy_files:
        print("\nFound NumPy files that might be using memory mapping:")
        for file in numpy_files:
            print(f"- {file} ({convert_size(os.path.getsize(file))})")
    
    # Check disk space
    if hasattr(os, 'statvfs'):  # Unix/Linux/Mac
        statvfs = os.statvfs(current_dir)
        free_space = statvfs.f_frsize * statvfs.f_bavail
        print(f"\nFree disk space: {convert_size(free_space)}")
    else:  # Windows
        import ctypes
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(current_dir), None, None, ctypes.pointer(free_bytes))
        print(f"\nFree disk space: {convert_size(free_bytes.value)}")
    
    clean_temp_files()