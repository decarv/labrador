#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./run.sh <file_name>"
    exit 1
fi

read -p "Executing file ${1}. Type Y to continue... " user_input
if [ "$user_input" != "Y" ]; then
    echo "Exiting..."
    exit 1
fi

source .venv/bin/activate

tmux kill-session -t labrador
tmux new-session -d -s labrador

tmux split-window -h
tmux send-keys "python3 ${1}" C-m
tmux select-pane -L
tmux split-window -v
tmux send-keys 'htop' C-m
tmux select-pane -D
tmux send-keys 'nvtop' C-m


# Attach to the tmux session
tmux attach -t experiments
