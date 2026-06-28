import os
import glob
import pandas as pd

def verify_datasets():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    print("==============================================")
    print("        PROJECT DATASET VERIFICATION          ")
    print("==============================================")
    print(f"Checking directory: {raw_dir}\n")
    
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data folder not found. Please run setup_project.py first.")
        return
        
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    if not csv_files:
        print("[X] Status: No datasets found in data/raw!")
        print("\nWhat you need to do:")
        print("1. Go to the Kaggle links provided:")
        print("   - Food Delivery: https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset")
        print("   - Weather Data: https://www.kaggle.com/datasets/hiteshsoneji/historical-weather-data-for-indian-cities")
        print("2. Download the datasets as zip files.")
        print("3. Extract them and place the CSV files inside:")
        print(f"   {raw_dir}")
        print("\nExpected files:")
        print("   - train.csv (or delivery dataset file)")
        print("   - bengaluru.csv, bombay.csv, delhi.csv, etc. (weather files)")
        return
        
    print(f"[OK] Found {len(csv_files)} CSV files in data/raw.\n")
    
    # Identify food delivery datasets
    delivery_files = [f for f in csv_files if 'delivery' in os.path.basename(f).lower() 
                      or 'train' in os.path.basename(f).lower() 
                      or 'test' in os.path.basename(f).lower()]
    
    # Filter weather files
    weather_files = [f for f in csv_files if f not in delivery_files]
    
    print("1. Food Delivery Dataset files:")
    if delivery_files:
        for f in delivery_files:
            fname = os.path.basename(f)
            try:
                df = pd.read_csv(f, nrows=5)
                print(f"   - {fname} (Shape: {df.shape[0]}x{len(df.columns)} verified, Columns: {list(df.columns)[:5]}...)")
            except Exception as e:
                print(f"   - [X] {fname} (Could not read: {e})")
    else:
        print("   - [X] No food delivery dataset found. Please upload 'train.csv'.")
        
    print("\n2. Weather Dataset files:")
    if weather_files:
        print(f"   - Found {len(weather_files)} weather files.")
        # Check one representative weather file
        rep_file = weather_files[0]
        fname = os.path.basename(rep_file)
        try:
            df = pd.read_csv(rep_file, nrows=5)
            print(f"   - Example: {fname} (Columns: {list(df.columns)})")
        except Exception as e:
            print(f"   - Example: [X] {fname} (Could not read: {e})")
    else:
        print("   - [X] No city weather files found. Please upload the historical weather CSV files.")
        
    print("\n==============================================")
    if delivery_files and weather_files:
        print("[OK] Status: All datasets verified! You are ready to train.")
        print("To preprocess the data, run:  python src/preprocess.py")
        print("To train the models, run:     python src/train.py")
    else:
        print("[!] Status: Missing required datasets. Please check the steps above.")
    print("==============================================")

if __name__ == '__main__':
    verify_datasets()
