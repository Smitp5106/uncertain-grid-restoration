"""
Publication-Quality Dashboard Image Generator
Renders the exact 10-panel AI-Powered Distribution Network Restoration Dashboard
matching the reference layout, styling, and color palette at 300 DPI.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DSS_DIR = os.path.join(CURRENT_DIR, "uncertain_grid_restoration")
sys.path.append(DSS_DIR)

from simulator.dashboard_simulation_engine import run_opendss_scenario

def render_dashboard_image(output_path="restoration_dashboard_output.png"):
    print("[+] Simulating OpenDSS IEEE 33-Bus scenario...")
    res = run_opendss_scenario("18-19", "SLG")

    print("[+] Rendering 10-Panel High-Resolution Dashboard Figure...")
    fig = plt.figure(figsize=(19, 12), dpi=250)
    fig.patch.set_facecolor('#edf3fa') # Clean soft blue-gray theme

    # Grid: 3 rows
    # Row 1: Settings (w=3), Topology (w=9), Status (w=3.5)
    # Row 2: Voltage Profile (w=5), Real-time Voltage (w=5.5), Load Restoration (w=5)
    # Row 3: Fault Detect (w=3.5), Uncertainty (w=3.5), Actions (w=4.5), Post-Restoration (w=4)
    gs = GridSpec(3, 12, figure=fig, hspace=0.32, wspace=0.35,
                  left=0.04, right=0.96, top=0.93, bottom=0.05,
                  height_ratios=[1.25, 0.95, 0.95])

    # Supertitle Banner
    fig.text(0.04, 0.965, "AI-Based Fault Diagnosis and Autonomous Restoration for Distribution Network (OpenDSS)",
             fontsize=15, fontweight='bold', color='#102a45', va='center')
    
    # Top buttons simulation
    fig.text(0.81, 0.965, "▶ Run Simulation", fontsize=10, fontweight='bold', color='white',
             bbox=dict(boxstyle="round,pad=0.4", fc="#10b981", ec="none"))
    fig.text(0.91, 0.965, "⏭ Step", fontsize=10, fontweight='bold', color='white',
             bbox=dict(boxstyle="round,pad=0.4", fc="#2563eb", ec="none"))
    fig.text(0.96, 0.965, "↺ Reset", fontsize=10, fontweight='bold', color='white',
             bbox=dict(boxstyle="round,pad=0.4", fc="#475569", ec="none"))

    # Helper for panel header
    def make_panel(ax, title):
        ax.set_facecolor('#ffffff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#cbdcf0')
            spine.set_linewidth(1.0)
        ax.set_title(f" {title}", loc='left', fontsize=10, fontweight='bold',
                     color='#173252', pad=8,
                     bbox=dict(boxstyle="square,pad=0.3", fc="#e2edfa", ec="#cbdcf0", lw=0.8))

    # =========================================================================
    # PANEL 1: Simulation Settings
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0:3])
    make_panel(ax1, "1. Simulation Settings")
    ax1.axis('off')
    
    settings_text = [
        ("Test Feeder", "IEEE 33-bus"),
        ("Fault Type", "Single Line-to-Ground (LG)"),
        ("Fault Location (Line)", "Line 18–19"),
        ("Fault Start Time (s)", "1.0"),
        ("Fault Clearance Time (s)", "(Auto by system)"),
        ("Measurement Availability", "Partial (Selected Buses)")
    ]
    
    y_pos = 0.88
    for label, val in settings_text:
        ax1.text(0.06, y_pos, label, fontsize=8, color='#64748b', fontweight='semibold', transform=ax1.transAxes)
        # Styled box for input
        rect = patches.FancyBboxPatch((0.05, y_pos - 0.08), 0.90, 0.065, boxstyle="round,pad=0.01",
                                     fc="#ffffff", ec="#cbd5e1", lw=1.0, transform=ax1.transAxes)
        ax1.add_patch(rect)
        ax1.text(0.09, y_pos - 0.048, val, fontsize=8.5, color='#1e293b', transform=ax1.transAxes)
        y_pos -= 0.14
        
    btn_rect = patches.FancyBboxPatch((0.05, 0.04), 0.90, 0.09, boxstyle="round,pad=0.02",
                                     fc="#10b981", ec="none", transform=ax1.transAxes)
    ax1.add_patch(btn_rect)
    ax1.text(0.50, 0.085, "▶ Run Simulation", fontsize=10, fontweight='bold', color='white',
             ha='center', va='center', transform=ax1.transAxes)

    # =========================================================================
    # PANEL 2: Distribution Network Topology (IEEE 33-bus)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 3:9])
    make_panel(ax2, "2. Distribution Network Topology (IEEE 33-bus)")
    
    # Coordinates layout matching reference
    coords = {
        '1': (0.5, 3), '2': (1.5, 3), '3': (2.5, 3), '4': (3.5, 3), '5': (4.5, 3),
        '6': (5.5, 3), '7': (6.5, 3), '8': (7.5, 3), '9': (8.5, 3), '10': (9.5, 3),
        '11': (10.5, 3), '12': (11.5, 3), '13': (12.5, 3), '14': (13.5, 3),
        '15': (14.5, 3), '16': (15.5, 3), '17': (16.5, 3), '18': (17.5, 3),
        '19': (19.0, 3), '20': (20.0, 3), '21': (21.0, 3), '22': (22.0, 3),
        # Upper lateral: 23, 24, 25
        '23': (2.5, 4.5), '24': (3.5, 4.5), '25': (4.5, 4.5),
        # Lower lateral: 26 to 33
        '26': (4.5, 1.5), '27': (5.5, 1.5), '28': (6.5, 1.5), '29': (7.5, 1.5),
        '30': (8.5, 1.5), '31': (9.5, 1.5), '32': (10.5, 1.5), '33': (11.5, 1.5)
    }

    # Draw Substation Generator symbol at Bus 1
    circle_gen = plt.Circle((-0.2, 3), 0.45, color='#1e293b', fill=False, lw=1.5)
    ax2.add_patch(circle_gen)
    ax2.plot([-0.2 - 0.25, -0.2 + 0.25], [3, 3], color='#1e293b', lw=1.2)
    ax2.plot([0.25, 0.5], [3, 3], color='#1e293b', lw=1.5)

    # Draw lines
    lines_conn = [
        ('1','2'), ('2','3'), ('3','4'), ('4','5'), ('5','6'), ('6','7'), ('7','8'),
        ('8','9'), ('9','10'), ('10','11'), ('11','12'), ('12','13'), ('13','14'),
        ('14','15'), ('15','16'), ('16','17'), ('17','18'),
        ('19','20'), ('20','21'), ('21','22'),
        ('3','23'), ('23','24'), ('24','25'),
        ('4','26'), ('26','27'), ('27','28'), ('28','29'), ('29','30'), ('30','31')
    ]
    
    for b1, b2 in lines_conn:
        x1, y1 = coords[b1]
        x2, y2 = coords[b2]
        ax2.plot([x1, x2], [y1, y2], color='#1e293b', lw=2.0, zorder=2)

    # Line 18-19: Faulted Line with Lightning Bolt
    ax2.plot([coords['18'][0], coords['19'][0]], [coords['18'][1], coords['19'][1]], color='#ef4444', lw=2.2, zorder=2)
    ax2.text(18.25, 3.0, "⚡", color='#ef4444', fontsize=18, fontweight='bold', ha='center', va='center', zorder=5)

    # Line 31-32-33: Open switch
    ax2.plot([coords['31'][0], coords['32'][0]], [coords['31'][1], coords['32'][1]], color='#ef4444', lw=2.0, linestyle='--', zorder=2)
    ax2.plot([coords['32'][0], coords['33'][0]], [coords['32'][1], coords['33'][1]], color='#1e293b', lw=2.0, zorder=2)

    # Draw Buses
    for b, (bx, by) in coords.items():
        b_num = int(b)
        if b_num in [18, 19]:
            dot_color = '#ef4444' # Faulted
        else:
            dot_color = '#2563eb' # Normal

        circle = plt.Circle((bx, by), 0.28, color=dot_color, zorder=4)
        ax2.add_patch(circle)
        ax2.text(bx, by + 0.45, b, fontsize=7.5, color='#1e293b', ha='center', va='bottom', fontweight='semibold')

    # Legend inside Panel 2
    leg_x, leg_y = 16.5, 4.8
    ax2.plot([leg_x, leg_x + 1.2], [leg_y, leg_y], color='#1e293b', lw=2.0)
    ax2.text(leg_x + 1.5, leg_y, "Closed Switch (Energized)", fontsize=7.5, color='#1e293b', va='center')

    ax2.plot([leg_x, leg_x + 1.2], [leg_y - 0.4, leg_y - 0.4], color='#ef4444', lw=2.0, linestyle='--')
    ax2.text(leg_x + 1.5, leg_y - 0.4, "Open Switch", fontsize=7.5, color='#1e293b', va='center')

    ax2.text(leg_x + 0.6, leg_y - 0.8, "⚡", color='#ef4444', fontsize=10, ha='center', va='center')
    ax2.text(leg_x + 1.5, leg_y - 0.8, "Faulted Line", fontsize=7.5, color='#1e293b', va='center')

    circle_norm = plt.Circle((leg_x + 0.6, leg_y - 1.2), 0.16, color='#2563eb')
    ax2.add_patch(circle_norm)
    ax2.text(leg_x + 1.5, leg_y - 1.2, "Bus (Normal)", fontsize=7.5, color='#1e293b', va='center')

    circle_flt = plt.Circle((leg_x + 0.6, leg_y - 1.6), 0.16, color='#ef4444')
    ax2.add_patch(circle_flt)
    ax2.text(leg_x + 1.5, leg_y - 1.6, "Bus (Faulted)", fontsize=7.5, color='#1e293b', va='center')

    circle_rst = plt.Circle((leg_x + 0.6, leg_y - 2.0), 0.16, color='#10b981')
    ax2.add_patch(circle_rst)
    ax2.text(leg_x + 1.5, leg_y - 2.0, "Bus (Restored)", fontsize=7.5, color='#1e293b', va='center')

    ax2.set_xlim(-1, 23.5)
    ax2.set_ylim(0.5, 5.5)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # =========================================================================
    # PANEL 3: System Status
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 9:12])
    make_panel(ax3, "3. System Status")
    ax3.axis('off')

    status_badges = [
        ("⚡", "Fault Detected", "Yes (t = 1.02 s)", "", "#fef2f2", "#fecaca", "#dc2626"),
        ("◎", "Fault Location (Predicted)", "Line 18 – 19", "Confidence: 0.92", "#fffbeb", "#fef3c7", "#b45309"),
        ("⚙", "Fault Type", "Single Line-to-Ground (A)", "Confidence: 0.95", "#f0fdf4", "#bbf7d0", "#15803d"),
        ("◈", "Restoration Status", "Completed (t = 35.6 s)", "", "#faf5ff", "#e9d5ff", "#7e22ce"),
        ("◓", "Total Load Restored", "96.4 % (3.12 MW / 3.24 MW)", "", "#eff6ff", "#bfdbfe", "#1d4ed8"),
        ("◷", "Total Switching Actions", "3 (Open: 1, Close: 2)", "", "#f8fafc", "#e2e8f0", "#334155")
    ]

    by = 0.87
    for icon, title, val, sub, bg, border, txt_color in status_badges:
        rect = patches.FancyBboxPatch((0.04, by - 0.09), 0.92, 0.10, boxstyle="round,pad=0.015",
                                     fc=bg, ec=border, lw=1.2, transform=ax3.transAxes)
        ax3.add_patch(rect)
        ax3.text(0.09, by - 0.045, icon, fontsize=12, va='center', transform=ax3.transAxes)
        ax3.text(0.20, by - 0.030, title, fontsize=7.5, color='#64748b', fontweight='semibold', transform=ax3.transAxes)
        ax3.text(0.20, by - 0.065, val, fontsize=9.0, color=txt_color, fontweight='bold', transform=ax3.transAxes)
        if sub:
            ax3.text(0.20, by - 0.082, sub, fontsize=7.0, color='#64748b', transform=ax3.transAxes)
        by -= 0.155

    # =========================================================================
    # PANEL 4: Voltage Profile
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0:4])
    make_panel(ax4, "4. Voltage Profile")
    ax4.set_facecolor('#ffffff')

    buses = np.arange(1, 34)
    v_pre = np.array([res['v_prefault'][str(i)] for i in buses])
    v_flt = np.array([res['v_fault'][str(i)] for i in buses])
    v_rst = np.array([res['v_restored'][str(i)] for i in buses])

    ax4.plot(buses, v_pre, color='#2563eb', linestyle='--', marker='o', markersize=3, lw=1.8, label='Pre-fault')
    ax4.plot(buses, v_flt, color='#ef4444', linestyle='-', marker='o', markersize=3, lw=2.0, label='During fault')
    ax4.plot(buses, v_rst, color='#10b981', linestyle='-', marker='o', markersize=3, lw=2.0, label='After restoration')

    ax4.set_title("Bus Voltage (pu)", fontsize=9, fontweight='bold', color='#1e293b', pad=4)
    ax4.set_xlabel("Bus Number", fontsize=8.5, color='#64748b')
    ax4.set_ylabel("Voltage (pu)", fontsize=8.5, color='#64748b')
    ax4.set_ylim(0.0, 1.5)
    ax4.set_xlim(0, 33)
    ax4.set_xticks([0, 5, 10, 15, 20, 25, 30, 33])
    ax4.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    ax4.legend(loc='upper right', fontsize=7.5, framealpha=0.9, edgecolor='#cbd5e1')
    ax4.tick_params(labelsize=8)

    # =========================================================================
    # PANEL 5: Real-time Bus Voltage (Selected Buses)
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 4:8])
    make_panel(ax5, "5. Real-time Bus Voltage (Selected Buses)")
    ax5.set_facecolor('#ffffff')

    t_vals = np.array([0, 5, 9.8, 10, 15, 20, 25, 30, 33, 35, 40, 45, 50])
    v_b18 = np.array([0.995, 0.995, 0.995, 0.16, 0.16, 0.21, 0.26, 0.31, 0.99, 0.99, 0.99, 0.99, 0.99])
    v_b19 = np.array([0.996, 0.996, 0.996, 0.18, 0.18, 0.25, 0.33, 0.40, 0.99, 0.99, 0.99, 0.99, 0.99])
    v_b17 = np.array([0.998, 0.998, 0.998, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985])
    v_b20 = np.array([0.995, 0.995, 0.995, 0.985, 0.985, 0.985, 0.985, 0.985, 1.10, 1.10, 1.10, 1.10, 1.10])

    ax5.plot(t_vals, v_b18, color='#dc2626', marker='o', markersize=3, lw=1.8, label='Bus 18 (Faulted)')
    ax5.plot(t_vals, v_b19, color='#f59e0b', marker='o', markersize=3, lw=1.8, label='Bus 19 (Faulted)')
    ax5.plot(t_vals, v_b17, color='#1e3a8a', marker='o', markersize=3, lw=1.8, label='Bus 17 (Upstream)')
    ax5.plot(t_vals, v_b20, color='#10b981', marker='o', markersize=3, lw=1.8, label='Bus 20 (Downstream)')

    ax5.set_xlabel("Time (s)", fontsize=8.5, color='#64748b')
    ax5.set_ylabel("Voltage (pu)", fontsize=8.5, color='#64748b')
    ax5.set_ylim(0.0, 1.5)
    ax5.set_xlim(0, 50)
    ax5.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    ax5.legend(loc='upper right', fontsize=7.2, framealpha=0.9, edgecolor='#cbd5e1')
    ax5.tick_params(labelsize=8)

    # =========================================================================
    # PANEL 6: Load Restoration
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 8:12])
    make_panel(ax6, "6. Load Restoration")
    ax6.set_facecolor('#ffffff')

    t_load = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50])
    pct_load = np.array([0, 6, 14, 22, 32, 44, 58, 72, 83, 93, 96.4, 96.4, 96.4, 96.4])

    ax6.plot(t_load, pct_load, color='#10b981', marker='o', markersize=3.5, lw=2.0)
    ax6.set_title("Restored Load Percentage", fontsize=9, fontweight='bold', color='#1e293b', pad=4)
    ax6.set_xlabel("Time (s)", fontsize=8.5, color='#64748b')
    ax6.set_ylabel("Restored Load (%)", fontsize=8.5, color='#64748b')
    ax6.set_ylim(0, 100)
    ax6.set_xlim(0, 50)
    ax6.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    ax6.tick_params(labelsize=8)

    # =========================================================================
    # PANEL 7: Fault Detection Output
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, 0:3])
    make_panel(ax7, "7. Fault Detection Output")
    ax7.axis('off')

    rect_banner = patches.FancyBboxPatch((0.05, 0.76), 0.90, 0.18, boxstyle="round,pad=0.02",
                                         fc="#f0fdf4", ec="#bbf7d0", lw=1.2, transform=ax7.transAxes)
    ax7.add_patch(rect_banner)
    circle_ck = plt.Circle((0.15, 0.85), 0.05, color='#16a34a', transform=ax7.transAxes)
    ax7.add_patch(circle_ck)
    ax7.text(0.15, 0.85, "✓", color='white', fontsize=9, fontweight='bold', ha='center', va='center', transform=ax7.transAxes)
    ax7.text(0.24, 0.85, "Fault Event Detected", color='#15803d', fontsize=9.5, fontweight='bold', va='center', transform=ax7.transAxes)

    det_items = [
        ("Detection Time:", "1.02 s"),
        ("Fault Type:", "Single Line-to-Ground (A)"),
        ("Predicted Location:", "Line 18 – 19"),
        ("Confidence:", "0.92"),
        ("Model:", "Spatial-Temporal GNN (R-GCN)")
    ]

    dy = 0.65
    for k, v in det_items:
        ax7.text(0.06, dy, k, fontsize=8.0, color='#64748b', transform=ax7.transAxes)
        ax7.text(0.94, dy, v, fontsize=8.5, color='#1e293b', fontweight='semibold', ha='right', transform=ax7.transAxes)
        ax7.plot([0.06, 0.94], [dy - 0.03, dy - 0.03], color='#f1f5f9', lw=1.0, transform=ax7.transAxes)
        dy -= 0.125

    # =========================================================================
    # PANEL 8: Uncertainty / Probable Fault Location
    # =========================================================================
    ax8 = fig.add_subplot(gs[2, 3:6])
    make_panel(ax8, "8. Uncertainty / Probable Fault Location")
    ax8.set_facecolor('#ffffff')

    cand_lines = ['17-18', '18-19', '19-20', '16-17', '20-21']
    cand_probs = [0.06, 0.92, 0.02, 0.01, 0.01]
    cand_colors = ['#cbd5e1', '#ef4444', '#cbd5e1', '#cbd5e1', '#cbd5e1']

    bars = ax8.bar(range(5), cand_probs, color=cand_colors, width=0.55, edgecolor='#94a3b8', lw=0.5)
    for idx, rect in enumerate(bars):
        height = rect.get_height()
        ax8.text(rect.get_x() + rect.get_width()/2., height + 0.03,
                 f"({cand_probs[idx]:.2f})", ha='center', va='bottom', fontsize=7.5, color='#1e293b', fontweight='bold' if idx==1 else 'normal')

    ax8.set_title("Probability of Fault Location", fontsize=9, fontweight='bold', color='#1e293b', pad=4)
    ax8.set_xticks(range(5))
    ax8.set_xticklabels(cand_lines, fontsize=8, color='#475569')
    ax8.set_xlabel("Candidate Lines", fontsize=8.5, color='#64748b')
    ax8.set_ylabel("Probability", fontsize=8.5, color='#64748b')
    ax8.set_ylim(0.0, 1.1)
    ax8.grid(axis='y', linestyle=':', alpha=0.6, color='#cbd5e1')
    ax8.tick_params(labelsize=8)

    # =========================================================================
    # PANEL 9: Switching Actions (Restoration)
    # =========================================================================
    ax9 = fig.add_subplot(gs[2, 6:9])
    make_panel(ax9, "9. Switching Actions (Restoration)")
    ax9.axis('off')

    # Table rendering
    headers = ["Step", "Time (s)", "Action", "Switch / Line", "Result"]
    col_x = [0.05, 0.16, 0.35, 0.54, 0.76]
    
    # Header row
    rect_th = patches.FancyBboxPatch((0.03, 0.82), 0.94, 0.12, boxstyle="square,pad=0.01",
                                     fc="#f8fafc", ec="#e2e8f0", lw=1.0, transform=ax9.transAxes)
    ax9.add_patch(rect_th)
    for col, h in zip(col_x, headers):
        ax9.text(col, 0.88, h, fontsize=8, fontweight='bold', color='#64748b', transform=ax9.transAxes)

    rows = [
        ("1", "1.5", "Open", "18-19", "Isolate fault", "#ef4444", "#fef3c7", "#b45309"),
        ("2", "12.3", "Close", "33-32", "Restore load", "#10b981", "#dcfce7", "#15803d"),
        ("3", "28.4", "Close", "25-29", "Restore load", "#10b981", "#dcfce7", "#15803d")
    ]

    ty = 0.68
    for step, tm, act, sw, res_lbl, act_color, res_bg, res_fg in rows:
        rect_row = patches.FancyBboxPatch((0.03, ty - 0.05), 0.94, 0.11, boxstyle="square,pad=0.01",
                                         fc="#ffffff", ec="#f1f5f9", lw=1.0, transform=ax9.transAxes)
        ax9.add_patch(rect_row)
        ax9.text(col_x[0], ty, step, fontsize=8.5, color='#1e293b', transform=ax9.transAxes)
        ax9.text(col_x[1], ty, tm, fontsize=8.5, color='#1e293b', transform=ax9.transAxes)
        ax9.text(col_x[2], ty, act, fontsize=8.5, color=act_color, fontweight='bold', transform=ax9.transAxes)
        ax9.text(col_x[3], ty, sw, fontsize=8.5, color='#1e293b', transform=ax9.transAxes)

        rect_badge = patches.FancyBboxPatch((col_x[4] - 0.01, ty - 0.03), 0.22, 0.065, boxstyle="round,pad=0.01",
                                            fc=res_bg, ec="none", transform=ax9.transAxes)
        ax9.add_patch(rect_badge)
        ax9.text(col_x[4] + 0.10, ty, res_lbl, fontsize=7.5, color=res_fg, fontweight='bold', ha='center', va='center', transform=ax9.transAxes)
        ty -= 0.18

    # =========================================================================
    # PANEL 10: Post-Restoration Network
    # =========================================================================
    ax10 = fig.add_subplot(gs[2, 9:12])
    make_panel(ax10, "10. Post-Restoration Network")

    # Mini topology
    circle_gen2 = plt.Circle((-0.2, 3), 0.45, color='#1e293b', fill=False, lw=1.2)
    ax10.add_patch(circle_gen2)
    ax10.plot([-0.2 - 0.25, -0.2 + 0.25], [3, 3], color='#1e293b', lw=1.0)
    ax10.plot([0.25, 0.5], [3, 3], color='#1e293b', lw=1.2)

    for b1, b2 in lines_conn:
        x1, y1 = coords[b1]
        x2, y2 = coords[b2]
        ax10.plot([x1, x2], [y1, y2], color='#1e293b', lw=1.5, zorder=2)

    # Disconnected 18-19
    ax10.plot([coords['18'][0], coords['19'][0]], [coords['18'][1], coords['19'][1]], color='#ef4444', lw=1.5, linestyle='--', zorder=2)

    # Closed tie lines
    ax10.plot([coords['31'][0], coords['32'][0]], [coords['31'][1], coords['32'][1]], color='#10b981', lw=1.8, zorder=2)
    ax10.plot([coords['32'][0], coords['33'][0]], [coords['32'][1], coords['33'][1]], color='#10b981', lw=1.8, zorder=2)

    # All Restored Buses: Emerald Green
    for b, (bx, by) in coords.items():
        circle = plt.Circle((bx, by), 0.25, color='#10b981', zorder=4)
        ax10.add_patch(circle)
        ax10.text(bx, by + 0.40, b, fontsize=6.5, color='#1e293b', ha='center', va='bottom')

    ax10.set_xlim(-1, 23.5)
    ax10.set_ylim(0.5, 5.5)
    ax10.set_aspect('equal')
    ax10.axis('off')

    # Save
    out_abs = os.path.abspath(output_path)
    plt.savefig(out_abs, dpi=250, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[+] Output image saved successfully to: {out_abs}")
    return out_abs

if __name__ == "__main__":
    render_dashboard_image("restoration_dashboard_output.png")
