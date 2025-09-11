import multiprocessing
import os

# Bind to Railway's PORT environment variable
port = os.environ.get("PORT", "8000")
bind = f"0.0.0.0:{port}"

# Workers - giới hạn cho Railway
cpu_count = multiprocessing.cpu_count()
calculated_workers = cpu_count * 2 + 1

if os.environ.get("RAILWAY_ENVIRONMENT"):
    workers = min(calculated_workers, 2)  # Max 2 workers trên Railway
    print(f"Railway: Using {workers} workers")
else:
    workers = min(calculated_workers, 4)  # Max 4 workers locally
    print(f"Local: Using {workers} workers")

worker_class = "sync"
worker_connections = 1000

# Timeout settings - QUAN TRỌNG
# API call có thể mất 90s + processing time
timeout = 120  # Tăng lên 120s để đủ cho API call 90s + processing
keepalive = 5
graceful_timeout = 30

# Memory management
max_requests = 300  # Giảm để tránh memory buildup từ AI calls
max_requests_jitter = 30

# Preload application
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True
enable_stdio_inheritance = True

# Process naming
proc_name = "studyapp_backend"

# Worker temp directory
worker_tmp_dir = "/tmp"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Railway specific
forwarded_allow_ips = "*"
secure_scheme_headers = {
    'X-FORWARDED-PROTOCOL': 'ssl',
    'X-FORWARDED-PROTO': 'https',
    'X-FORWARDED-SSL': 'on'
}

# Log configuration
print(f"Gunicorn configuration:")
print(f"  Workers: {workers}")
print(f"  Timeout: {timeout}s (for AI API calls)")
print(f"  Max requests per worker: {max_requests}")
print(f"  Environment: {'Railway' if os.environ.get('RAILWAY_ENVIRONMENT') else 'Local'}")
