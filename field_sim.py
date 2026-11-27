"""Virtual 3x3 field: a simulated sprayer, Mode A (spray everything) vs Mode B (AI-targeted).

ALL numbers below are ASSUMPTIONS for a simulation - show them on screen as such.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOSE_FACTOR = {"Mild": 1.0, "Moderate": 1.0, "Severe": 1.0}  # assumption: sprayed squares get the full label dose; we never change concentration; severity is for priority only
COLORS = {"infected": "#e06666", "retake": "#ffd966", "healthy": "#93c47d"}


def snake_route(rows=3, cols=3):
    """Left-to-right on row 0, right-to-left on row 1, and so on."""
    return [(r, c if r % 2 == 0 else cols - 1 - c) for r in range(rows) for c in range(cols)]


def simulate(grid, mode, sec_move=10, sec_spray=15, litres_per_cell=0.5, buffer=False):
    """grid[r][c] = ("infected", "Mild|Moderate|Severe") | ("healthy", None) | ("retake", None)."""
    rows, cols = len(grid), len(grid[0])
    route = snake_route(rows, cols)
    # buffer: in Mode B, also spray orthogonal neighbours of infected cells
    buffer_cells = []
    if buffer and mode == "B":
        infected = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c][0] == "infected"}
        for (r, c) in infected:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc][0] != "infected":
                    if (nr, nc) not in buffer_cells:
                        buffer_cells.append((nr, nc))
    nozzle, litres, seconds = [], 0.0, 0
    for r, c in route:
        status, level = grid[r][c]
        spray = (mode == "A") or (mode == "B" and (status == "infected" or (buffer and (r, c) in buffer_cells)))
        seconds += sec_move + (sec_spray if spray else 0)
        if spray:
            litres += litres_per_cell * (1.0 if mode == "A" else DOSE_FACTOR[level if level else "Moderate"])
        nozzle.append(spray)
    return {"mode": mode, "route": route, "nozzle_on": nozzle,
            "seconds": seconds, "litres": round(litres, 2),
            "retake_cells": [(r, c) for r, c in route if grid[r][c][0] == "retake"],
            "buffer_cells": buffer_cells}


def compare(grid, buffer=False, **kw):
    a = simulate(grid, "A", **kw)
    b = simulate(grid, "B", buffer=buffer, **kw)
    return {"A": a, "B": b,
            "litres_saved_pct": round(100 * (a["litres"] - b["litres"]) / a["litres"], 1),
            "time_saved_pct": round(100 * (a["seconds"] - b["seconds"]) / a["seconds"], 1),
            "buffer_cells": b.get("buffer_cells", [])}


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
