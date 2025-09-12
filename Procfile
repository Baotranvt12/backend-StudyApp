web: gunicorn backend.wsgi:application \
  --bind 0.0.0.0:$PORT \
  --workers ${WEB_CONCURRENCY:-2} \
  --threads ${GTHREADS:-8} \
  --worker-class gthread \
  --timeout 120 --keep-alive 5 --graceful-timeout 20 \
  --max-requests 300 --max-requests-jitter 60 \
  --access-logfile - --log-level info
