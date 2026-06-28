import os

def create_directory_structure():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directories = [
        os.path.join(base_dir, 'data', 'raw'),
        os.path.join(base_dir, 'data', 'processed'),
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'notebooks'),
        os.path.join(base_dir, 'src')
    ]
    
    print("Creating project directories...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created: {directory}")
        else:
            print(f"Already exists: {directory}")
            
    # Create empty __init__.py in src/ if it doesn't exist
    init_py = os.path.join(base_dir, 'src', '__init__.py')
    if not os.path.exists(init_py):
        with open(init_py, 'w') as f:
            pass
        print(f"Created init file: {init_py}")

    print("\nProject directories setup successfully!")
    print("Next steps:")
    print("1. Download the datasets from Kaggle.")
    print("2. Put the downloaded CSV files inside the folder:")
    print(f"   {os.path.join(base_dir, 'data', 'raw')}")

if __name__ == '__main__':
    create_directory_structure()
