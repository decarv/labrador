#!/bin/bash

source .venv/bin/activate

# Create a new tmux session without attaching to it
tmux kill-session -t experiments
tmux new-session -d -s experiments

# Split the window horizontally
tmux split-window -h
# Run nvtop in the left pane
tmux send-keys 'nvtop' C-m

# Split the right window vertically
tmux select-pane -R
tmux split-window -v
# Run htop in the upper right pane
tmux send-keys 'htop' C-m

# Switch to the lower right pane
tmux select-pane -D
# Run the experiments.py script
tmux send-keys 'python3 ./src/experiments.py' C-m

# Finally, attach to the tmux session
tmux attach -t experiments
