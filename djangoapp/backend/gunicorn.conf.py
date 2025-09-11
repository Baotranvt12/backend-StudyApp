# gunicorn.conf.py
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100

# Timeout settings - QUAN TRỌNG để tránh worker timeout
timeout = 120  # Tăng từ 30s mặc định lên 120s
keepalive = 5
graceful_timeout = 30

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Preload application
preload_app = True

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "studyapp_backend"

# Worker temp directory (use shared memory for better performance)
worker_tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None

# Daemon mode 
daemon = False

# PID file
pidfile = "/tmp/gunicorn_studyapp.pid"

# User and group 
# user = "www-data"
# group = "www-data"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
