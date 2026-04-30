import sys, traceback
sys.path.insert(0, r'E:\bianchen\bysj')
try:
    from core.plate_recognizer import PlateRecognizer
    print("PlateRecognizer imported successfully")
    pr = PlateRecognizer()
    print("PlateRecognizer initialized successfully!")
except Exception as e:
    print("ERROR:", type(e).__name__, str(e))
    traceback.print_exc()
