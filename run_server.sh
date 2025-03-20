#!/bin/bash

# Number of worker processes (adjust based on your CPU cores)
WORKERS=4

# Start Gunicorn with eventlet worker
exec gunicorn --worker-class eventlet \
              --workers $WORKERS \
              --bind 0.0.0.0:8000 \
              --timeout 120 \
              --access-logfile - \
              --error-logfile - \
              --log-level info \
              "app:app"