"""
AI-Powered Distribution Network Restoration Dashboard Runner (Root Shortcut)
"""
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sub_dir = os.path.join(root_dir, "uncertain_grid_restoration")

if sub_dir not in sys.path:
    sys.path.insert(0, sub_dir)

import run_dashboard as _rd
DashboardHandler = _rd.DashboardHandler
main = _rd.main

if __name__ == "__main__":
    main()
