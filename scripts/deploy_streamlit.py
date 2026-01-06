#!/usr/bin/env python3
"""Deployment helper for Streamlit Cloud. Author: Tharun Ponnam"""

import argparse
import subprocess
from pathlib import Path

def check_requirements():
    required = ["app.py", "requirements.txt", "README.md", ".streamlit/config.toml"]
    return [f for f in required if not Path(f).exists()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    
    if args.check:
        missing = check_requirements()
        if missing:
            print(f"Missing: {missing}")
        else:
            print("✅ Ready for Streamlit Cloud!")
            print("\nSteps:")
            print("1. Push to GitHub")
            print("2. Go to streamlit.io/cloud")
            print("3. Select repository and app.py")
            print("4. Deploy!")
    elif args.local:
        subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
