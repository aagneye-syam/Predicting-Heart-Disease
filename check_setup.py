"""
Setup Checker - Verify everything is ready
Run this to check if your environment is properly configured
"""

import sys
import os

print("=" * 70)
print("HEART DISEASE PREDICTION - SETUP CHECKER")
print("=" * 70)

all_good = True

# Check Python version
print("\n[1] Checking Python version...")
version = sys.version_info
if version.major == 3 and version.minor >= 8:
    print(f"    [OK] Python {version.major}.{version.minor}.{version.micro}")
else:
    print(f"    [WARNING] Python {version.major}.{version.minor} - Recommend 3.8+")
    all_good = False

# Check required packages
print("\n[2] Checking required packages...")
required_packages = {
    'pandas': 'Data manipulation',
    'numpy': 'Numerical operations',
    'sklearn': 'Machine learning (scikit-learn)'
}

for package, description in required_packages.items():
    try:
        __import__(package)
        print(f"    [OK] {package:15} - {description}")
    except ImportError:
        print(f"    [MISSING] {package:15} - {description}")
        print(f"              Install with: pip install {package}")
        all_good = False

# Check data files
print("\n[3] Checking data files...")
data_files = ['data/train.csv', 'data/test.csv', 'data/sample_submission.csv']

for file in data_files:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"    [OK] {file:30} ({size_mb:.2f} MB)")
    else:
        print(f"    [MISSING] {file}")
        all_good = False

# Check script files
print("\n[4] Checking script files...")
script_files = [
    'train_model.py',
    'train_model_advanced.py',
    'compare_models.py'
]

for file in script_files:
    if os.path.exists(file):
        print(f"    [OK] {file}")
    else:
        print(f"    [MISSING] {file}")
        all_good = False

# Check if submission exists
print("\n[5] Checking submission file...")
if os.path.exists('submission.csv'):
    size_mb = os.path.getsize('submission.csv') / (1024 * 1024)
    print(f"    [OK] submission.csv exists ({size_mb:.2f} MB)")
    print(f"         You can submit this to Kaggle now!")
else:
    print(f"    [INFO] submission.csv not found")
    print(f"           Run 'python train_model.py' to create it")

# Final summary
print("\n" + "=" * 70)
if all_good:
    print("SUCCESS! Everything is set up correctly!")
    print("=" * 70)
    print("\nYou're ready to go! Next steps:")
    print("1. Run: python train_model.py")
    print("2. Upload submission.csv to Kaggle")
    print("3. Check your score on the leaderboard")
else:
    print("SETUP INCOMPLETE - Please fix the issues above")
    print("=" * 70)
    print("\nCommon fixes:")
    print("1. Install packages: pip install -r requirements.txt")
    print("2. Make sure data files are in the 'data' folder")
    print("3. Check that you're in the correct directory")

print("\n" + "=" * 70)
