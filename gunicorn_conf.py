bind = "0.0.0.0:8000"
workers = 1  # start conservatively
worker_class = "gthread"
threads = 2
timeout = 600  # allow model load and inference
preload_app = True
max_requests = 100
max_requests_jitter = 10
