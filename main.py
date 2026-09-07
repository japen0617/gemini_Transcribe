import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)] + sys.argv[1:]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
