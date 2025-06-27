#!/usr/bin/env bash
# replace localhost with the port you see on the smartphone
export ROS_MASTER_URI="http://10.15.2.184:11311"

# You want your local IP, usually starting with 192.168, following RFC1918
# Windows powershell:
#    (Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress
# linux:
#    hostname -I | awk '{print $1}'
# macOS:
#    ifconfig en0 | awk '/inet / {print $2}'
export COPPELIA_SIM_IP="130.37.223.182"
# MacOS
# zsh ./scripts/start_coppelia_sim.zsh ./scenes/arena_push_simp.ttt 23000 -h
# zsh ./scripts/run_apple_sillicon.zsh --simulation