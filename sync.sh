#!/bin/bash
# Sync local code to GCP instance (excludes data, git, and caches)
rsync -avz --exclude '.git' --exclude 'data' --exclude '__pycache__' \
  --exclude '*.pyc' --exclude 'logs/' --exclude '*.egg-info' \
  ~/UCLA/CS147A/emg2qwerty/ emg2qwerty-dev:~/emg2qwerty/
