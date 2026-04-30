import subprocess, sys, os

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "app", "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path] + sys.argv[1:])
