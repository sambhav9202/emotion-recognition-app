import threading
import requests
import time

def keep_alive():
    while True:
        try:
            requests.get('https://emotion-recognition-app-v6rx.onrender.com/health')
        except:
            pass
        time.sleep(840)  # ping every 14 minutes

t = threading.Thread(target=keep_alive)
t.daemon = True
t.start()
