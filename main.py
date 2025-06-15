from flask import Flask, request, jsonify, Response
import pandas as pd
import os
import loader.stockdata as dl
import time
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import threading
import sys
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import configparser
import atexit
import signal
from time import sleep
import tempfile

# Shit windows falls ich Geld habe nutze ich einfach linux@@@
try:
    import fcntl
except ImportError:
    fcntl = None
# Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logging.getLogger('apscheduler').setLevel(logging.WARNING)

app = Flask(__name__)

# Global variables for cleanup
scheduler = None
file_observer = None
shutdown_in_progress = False
restart_count = 0
MAX_RESTART_COUNT = 5  # Prevent infinite restart loops

# Configuration
config = configparser.ConfigParser()
config.read('config.ini')

AUTO_RESTART = config.getboolean('DEFAULT', 'auto_restart', fallback=True)
AUTO_UPDATE = config.getboolean('DEFAULT', 'auto_update', fallback=True)

logging.info(f"Configuration loaded: auto_restart={AUTO_RESTART}, auto_update={AUTO_UPDATE}")

# Commands
def listen_for_update():
    try:
        while True:
            try:
                cmd = input()
                match cmd.strip():
                    case ':update':
                        print('Updating all stock data ...')
                        try:
                            dl.update_all()
                            print('Update finished.')
                        except Exception as e:
                            print(f'Error: {e}')
                    case ':help':
                        print('Available commands:')
                        print(':update - Update all stock data')
                        print(':help - Show this help message')
                    case _:
                        print(f'Unknown command: {cmd}')
                        print('Type :help for available commands')
            except EOFError:
                logging.info("Interactive input not available, command thread exiting.")
                sys.exit(0)
                break
            except KeyboardInterrupt:
                logging.info("Command input thread interrupted.")
                break
    except Exception as e:
        logging.error(f"Error in command listener: {e}")

def startup_tasks():
    if AUTO_UPDATE:
        for task in SCHEDULED_TASKS:
            logging.info(f"Running startup task: {task['name']}")
            try:
                task['func']()
            except Exception as e:
                logging.error(f"Error in startup task {task['name']}: {e}")
    else:
        logging.info("Auto-update is disabled. Skipping startup tasks.")

SCHEDULED_TASKS = [
    {
        'name': 'update_all_daily',
        'func': dl.update_all,
        'trigger': CronTrigger(hour=0, minute=0, second=0),
    },
]

def setup_scheduler():
    global scheduler
    if AUTO_UPDATE:
        if scheduler is not None:
            try:
                scheduler.shutdown(wait=False)
                logging.info("Scheduler has been shutted down") 
            except Exception as e:
                logging.warning(f"Error shutting down previous scheduler: {e}")
        
        scheduler = BackgroundScheduler()
        scheduler._logger.setLevel(logging.WARNING)
        
        for task in SCHEDULED_TASKS:
            scheduler.add_job(task['func'], task['trigger'], id=task['name'], replace_existing=True)
        
        scheduler.start()
        logging.info("Scheduler started with all tasks.")
        return scheduler
    else:
        logging.info("Auto-update is disabled. Scheduler not started.")
        return None

# Response time
@app.before_request
def start_timer():
    request.start_time = time.time()
def print_response_time(response):
    if hasattr(request, 'start_time'):
        elapsed = time.time() - request.start_time
        response.headers['X-Response-Time'] = f"{elapsed:.3f}s"
        logging.info(f"Response time: {elapsed:.3f}s")
    return response
app.after_request(print_response_time)

# Api
@app.route('/api/stock/data/<ticker>', methods=['GET'])
def read_stock_data(ticker):
    try:
        filepath = f"data/{ticker}.csv"
        if not os.path.exists(filepath):
            dl.load(ticker)
            if not os.path.exists(filepath):
                return jsonify({"error": "Stock data not found"}), 404
        else:
            dl.load(ticker)
        stock_df = pd.read_csv(filepath)
        stock_df['index'] = pd.to_datetime(stock_df['index'], utc=True)
        stock_df['index'] = stock_df['index'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        return jsonify(stock_df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/data/<ticker>/<time>', methods=['GET'])
def real_time_stock_data(ticker, time):
    json_data = dl.update(ticker, time)
    if json_data is None:
        return jsonify({"error": "Stock data not found"}), 404
    return jsonify(json_data)

@app.route('/api/stock/data/update/<ticker>/<time>', methods=['GET'])
def update_stock_data(ticker, time):
    json_data = dl.read(ticker, time)
    if json_data is None:
        return jsonify({"error": "Stock data not found"}), 404
    return jsonify(json_data)

@app.route('/.well-known/appspecific/com.chrome.devtools.json')
def chrome_devtools_dummy():
    return jsonify({"message": ""})

class FileWatcher(FileSystemEventHandler):
    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_modified = {}
        self.restart_scheduled = False
        self.debounce_time = 5  # Increased debounce time
        super().__init__()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith('.py'):
            return
        if 'phi_env' in event.src_path or '__pycache__' in event.src_path:
            return
        if self.restart_scheduled:
            return
            
        # Skip if this is main.py and we're already in a restart process
        if event.src_path.endswith('main.py') and shutdown_in_progress:
            return
            
        current_time = time.time()
        if event.src_path in self.last_modified:
            if current_time - self.last_modified[event.src_path] < self.debounce_time:
                return
        self.last_modified[event.src_path] = current_time
        self.restart_scheduled = True
        logging.info(f"File {event.src_path} was modified. Scheduling restart in 2 seconds...")
        threading.Timer(2.0, self.restart_callback).start()

def restart_application():
    logging.info("Restarting application...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    venv_python = os.path.join(project_root, 'phi_env', 'Scripts', 'python.exe')
    
    if not os.path.exists(venv_python):
        venv_python = sys.executable
        logging.warning(f"Virtual environment python not found, using {sys.executable}")
    
    script_path = os.path.join(project_root, 'main.py')
    
    logging.info(f"Restarting with executable: {venv_python}")
    logging.info(f"Script path: {script_path}")
    logging.info(f"Working directory: {os.getcwd()}")
    
    global restart_count
    restart_count += 1
    if restart_count > MAX_RESTART_COUNT:
        logging.error(f"Max restart limit reached ({MAX_RESTART_COUNT}). Aborting restart.")
        return
    cleanup_resources()
    subprocess.Popen([venv_python, script_path], cwd=project_root)
    os._exit(0)

def setup_file_watcher():
    global file_observer
    if AUTO_RESTART:
        try:
            if file_observer is not None:
                try:
                    file_observer.stop()
                    file_observer.join(timeout=2)
                except Exception as e:
                    logging.warning(f"Error stopping previous file observer: {e}")
            
            event_handler = FileWatcher(restart_application)
            file_observer = Observer()
            file_observer.schedule(event_handler, path='.', recursive=True)
            file_observer.start()
            logging.info("File watcher started.")
            return file_observer
        except Exception as e:
            logging.error(f"Failed to setup file watcher: {e}")
            return None
    else:
        logging.info("Auto-restart is disabled. File watcher not started.")
        return None

def cleanup_resources():
    global scheduler, file_observer, shutdown_in_progress
    
    if shutdown_in_progress:
        return
    shutdown_in_progress = True
    logging.info("Cleaning up resources...")
    try:
        if file_observer:
            logging.info("Stopping file observer...")
            file_observer.stop()
            file_observer.join(timeout=2)
    except Exception as e:
        logging.warning(f"Error stopping file observer: {e}")
    
    try:
        if scheduler and scheduler.running:
            logging.info("Shutting down scheduler...")
            scheduler.shutdown(wait=False)
    except Exception as e:
        logging.warning(f"Error shutting down scheduler: {e}")
    release_lock()
    
    scheduler = None
    file_observer = None
    logging.info("Resources cleaned up.")

def signal_handler(signum, frame):
    global shutdown_in_progress
    
    if shutdown_in_progress:
        return
    
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    cleanup_resources()

LOCK_FILE = os.path.join(tempfile.gettempdir(), 'finanz_app.lock')
lock_file_handle = None

def acquire_lock():
    global lock_file_handle
    try:
        lock_file_handle = open(LOCK_FILE, 'w')
        # fuck windows ich muss einen anderen Weg finden
        if os.name == 'nt':  # Windows
            import msvcrt
            try:
                msvcrt.locking(lock_file_handle.fileno(), msvcrt.LK_NBLCK, 1)
                lock_file_handle.write(str(os.getpid()))
                lock_file_handle.flush()
                return True
            except IOError:
                return False
        else:  # Unix/Linux
            if fcntl:
                fcntl.flock(lock_file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_file_handle.write(str(os.getpid()))
                lock_file_handle.flush()
                return True
            else:
                if os.path.exists(LOCK_FILE):
                    return False
                lock_file_handle.write(str(os.getpid()))
                lock_file_handle.flush()
                return True
    except (IOError, OSError):
        if lock_file_handle:
            lock_file_handle.close()
        return False

def release_lock():
    """Release the file lock"""
    global lock_file_handle
    if lock_file_handle:
        try:
            if os.name == 'nt':  # Windows
                import msvcrt
                msvcrt.locking(lock_file_handle.fileno(), msvcrt.LK_UNLCK, 1)
            lock_file_handle.close()
            if os.path.exists(LOCK_FILE):
                os.unlink(LOCK_FILE)
        except (IOError, OSError) as e:
            logging.warning(f"Error releasing lock: {e}")
        lock_file_handle = None

# Main
if __name__ == "__main__":
    if not acquire_lock():
        logging.error("Another instance of the application is already running.")
        sys.exit(1)
    atexit.register(cleanup_resources)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if sys.stdin.isatty():
        threading.Thread(target=listen_for_update, daemon=True).start()
        logging.info("Interactive command listener started. Type :help for commands.")
    else:
        logging.info("Non-interactive mode detected. Command listener disabled.")    
    scheduler = setup_scheduler()
    startup_tasks()
    file_observer = setup_file_watcher()
    
    try:
        # Disable Flucks debug mode and reloader cause i will use my reloader
        app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)
    except KeyboardInterrupt:
        logging.info("Application interrupted by user.")
    except Exception as e:
        logging.error(f"Application error: {e}")
    finally:
        cleanup_resources()
        release_lock()