"""Virtual 3x3 field: a simulated sprayer, Mode A (spray everything) vs Mode B (AI-targeted).

ALL numbers below are ASSUMPTIONS for a simulation - show them on screen as such.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOSE_FACTOR = {"Mild": 0.6, "Moderate": 1.0, "Severe": 1.4}  # assumption: more dose for worse cells
COLORS = {"infected": "#e06666", "retake": "#ffd966", "healthy": "#93c47d"}


def snake_route(rows=3, cols=3):
    """Left-to-right on row 0, right-to-left on row 1, and so on."""
    return [(r, c if r % 2 == 0 else cols - 1 - c) for r in range(rows) for c in range(cols)]


def simulate(grid, mode, sec_move=10, sec_spray=15, litres_per_cell=0.5):
    """grid[r][c] = ("infected", "Mild|Moderate|Severe") | ("healthy", None) | ("retake", None)."""
    route = snake_route(len(grid), len(grid[0]))
    nozzle, litres, seconds = [], 0.0, 0
    for r, c in route:
        status, level = grid[r][c]
        spray = (mode == "A") or (mode == "B" and status == "infected")
        seconds += sec_move + (sec_spray if spray else 0)
        if spray:
            litres += litres_per_cell * (1.0 if mode == "A" else DOSE_FACTOR[level])
        nozzle.append(spray)
    return {"mode": mode, "route": route, "nozzle_on": nozzle,
            "seconds": seconds, "litres": round(litres, 2),
            "retake_cells": [(r, c) for r, c in route if grid[r][c][0] == "retake"]}


def compare(grid, **kw):
    a, b = simulate(grid, "A", **kw), simulate(grid, "B", **kw)
    return {"A": a, "B": b,
            "litres_saved_pct": round(100 * (a["litres"] - b["litres"]) / a["litres"], 1),
            "time_saved_pct": round(100 * (a["seconds"] - b["seconds"]) / a["seconds"], 1)}


def draw(grid, result, title=""):
    fig, ax = plt.subplots(figsize=(4, 4))
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            status, level = grid[r][c]
            ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1, fc=COLORS[status], ec="black"))
            ax.text(c + .5, rows - 1 - r + .8, status + (f"\n{level}" if level else ""), ha="center", va="center", fontsize=8)
    pts = [(c + .5, rows - 1 - r + .5) for r, c in result["route"]]
    for i in range(len(pts) - 1):
        on = result["nozzle_on"][i]
        ax.annotate("", xy=pts[i + 1], xytext=pts[i],
                    arrowprops=dict(arrowstyle="->", color="blue" if on else "gray", lw=2 if on else 1, ls="-" if on else "--"))
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or f"Mode {result['mode']}: {result['litres']} L, {result['seconds']} s", fontsize=10)
    return fig


EXAMPLE_GRID = [
    [("infected", "Severe"), ("healthy", None), ("healthy", None)],
    [("healthy", None), ("infected", "Moderate"), ("retake", None)],
    [("retake", None), ("healthy", None), ("infected", "Mild")],
]

if __name__ == "__main__":
    out = compare(EXAMPLE_GRID)
    print("Mode A:", out["A"]["litres"], "L,", out["A"]["seconds"], "s")
    print("Mode B:", out["B"]["litres"], "L,", out["B"]["seconds"], "s, retake:", out["B"]["retake_cells"])
    print("Litres saved:", out["litres_saved_pct"], "%  | Time saved:", out["time_saved_pct"], "%")
    draw(EXAMPLE_GRID, out["B"]).savefig("field_mode_B.png", dpi=120)
