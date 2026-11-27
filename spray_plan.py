"""Build a spray plan from field-map cells and a field_sim result."""
from weather_rules import check_weather


def _priority(level):
    if level == "Severe":
        return "Urgent"
    if level == "Moderate":
        return "Soon"
    if level == "Mild":
        return "Monitor"
    return None


def build_plan(cells, result, price_per_litre=None, weather=None):
    # map (r,c) -> cell
    by_pos = {(c["row"], c["col"]): c for c in cells}
    route = result["B"]["route"]
    nozzle = result["B"]["nozzle_on"]

    # route order and nozzle per cell
    route_order = {}
    nozzle_map = {}
    for i, (r, c) in enumerate(route):
        name = None
        # find cell name
        for cell in cells:
            if cell["row"] == r and cell["col"] == c:
                name = cell["name"]
                break
        if name:
            route_order[name] = i + 1
            nozzle_map[name] = bool(nozzle[i])

    per_square = []
    for cell in cells:
        name = cell["name"]
        status = cell["status"]
        if status == "infected":
            action = "Spray"
        elif status == "healthy":
            action = "Skip"
        else:
            action = "Retake"
        prio = _priority(cell.get("level")) if status == "infected" else None
        per_square.append({
            "cell": name,
            "action": action,
            "priority": prio if prio else "-",
            "route_order": route_order.get(name),
            "nozzle": "ON" if nozzle_map.get(name) else "OFF",
            "disease": cell.get("class"),
            "confidence": cell.get("confidence"),
            "severity": cell.get("severity"),
        })

    sprayed = sum(1 for p in per_square if p["action"] == "Spray")
    total = len(per_square)
    pct_sprayed = round(100.0 * sprayed / total, 1) if total else 0.0

    litres_a = result["A"]["litres"]
    litres_b = result["B"]["litres"]
    litres_saved = round(litres_a - litres_b, 2)
    time_saved = result["A"]["seconds"] - result["B"]["seconds"]
    money_saved = None
    if price_per_litre is not None:
        money_saved = round(litres_saved * price_per_litre, 2)

    steps = []
    for i, (r, c) in enumerate(route):
        name = None
        for cell in cells:
            if cell["row"] == r and cell["col"] == c:
                name = cell["name"]
                break
        if name is None:
            continue
        prio = None
        for p in per_square:
            if p["cell"] == name:
                prio = p["priority"]
                break
        on = nozzle[i]
        step = f"{i+1}. Go to {name} - nozzle {'ON' if on else 'OFF'}"
        if on and prio and prio != "-":
            step += f" ({prio})"
        steps.append(step)

    retake_n = sum(1 for p in per_square if p["action"] == "Retake")
    if retake_n > 0:
        headline = f"Retake {retake_n} photos first"
    elif sprayed > 0:
        headline = f"Spray {sprayed} of {total} sections"
    else:
        headline = "No spraying needed"

    # gate
    if weather is not None:
        t = weather.get("temp_c", 28.0)
        h = weather.get("humidity_pct", 55.0)
        wn = weather.get("wind_kmh", 8.0)
        rn = weather.get("rain_pct", 10.0)
        wres = check_weather(t, h, wn, rn)
        if wres["decision"] == "WAIT":
            gate = "WAIT"
        else:
            gate = "OK to spray (check the label)"
        gate_reasons = [r["text"] for r in wres["why"]] + wres.get("notes", [])
    else:
        gate = "No weather data"
        gate_reasons = []

    totals = {
        "squares_sprayed": sprayed,
        "pct_field_sprayed": pct_sprayed,
        "litres_mode_a": litres_a,
        "litres_mode_b": litres_b,
        "litres_saved": litres_saved,
        "time_saved_seconds": time_saved,
        "time_saved_pct": result.get("time_saved_pct"),
        "money_saved": money_saved,
    }

    return {
        "per_square": per_square,
        "totals": totals,
        "steps": steps,
        "headline": headline,
        "gate": gate,
        "gate_reasons": gate_reasons,
    }
