#!/bin/bash

# Start a new tmux session in detached mode
tmux new-session -d

# Split the window vertically into two panes
tmux split-window -h

# Split the left pane horizontally into three
tmux select-pane -t 0
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# Split the right pane horizontally into three
tmux select-pane -t 3
tmux split-window -v
tmux select-pane -t 3
tmux split-window -v

# Attach to the tmux session
tmux attach-session

