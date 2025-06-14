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
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import configparser

# Config
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Configuration
config = configparser.ConfigParser()
config.read('config.ini')

AUTO_RESTART = config.getboolean('DEFAULT', 'auto_restart', fallback=True)
AUTO_UPDATE = config.getboolean('DEFAULT', 'auto_update', fallback=True)

logging.info(f"Configuration loaded: auto_restart={AUTO_RESTART}, auto_update={AUTO_UPDATE}")

# Commands
def listen_for_update():
    while True:
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

# Startup and tasks to run on startup
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
    if AUTO_UPDATE:
        scheduler = BackgroundScheduler()
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


# File watcher for auto-restart; avoid auto restart of flask, because it will cause loop
class FileWatcher(FileSystemEventHandler):
    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_modified = {}
        super().__init__()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if not event.src_path.endswith('.py'):
            return
        
        current_time = time.time()
        if event.src_path in self.last_modified:
            if current_time - self.last_modified[event.src_path] < 2:
                return
        
        self.last_modified[event.src_path] = current_time
        
        logging.info(f"File {event.src_path} was modified. Restarting...")
        self.restart_callback()

def restart_application():
    logging.info("Restarting application...")
    os.execv(sys.executable, ['python'] + sys.argv)

def setup_file_watcher():
    if AUTO_RESTART:
        try:
            event_handler = FileWatcher(restart_application)
            observer = Observer()
            observer.schedule(event_handler, path='.', recursive=True)
            observer.start()
            logging.info("Application will restart when Python files are modified.")
            return observer
        except Exception as e:
            logging.error(f"Failed to setup file watcher: {e}")
            return None
    else:
        logging.info("Auto-restart is disabled. File watcher not started.")
        return None

# Main
if __name__ == "__main__":
    threading.Thread(target=listen_for_update, daemon=True).start()
    scheduler = setup_scheduler()
    startup_tasks()
    file_observer = setup_file_watcher()
    try:
        app.run(debug=True, use_reloader=False)
    finally:
        if file_observer:
            file_observer.stop()
            file_observer.join()
        if scheduler:
            scheduler.shutdown()