"""Weather check. Every rule carries its source. LABEL INSTRUCTIONS ALWAYS WIN.

Rules come from US/Canadian extension guidance (no Indian equivalent was found
in our search) - say so, and replace with your product label / ICAR / state
agriculture department values when you have them.
"""
RULES = [
    {"id": "wind", "text": "Do not spray if wind is above 10 mph (16 km/h).",
     "source": "Virginia Cooperative Extension ENTO-453", "url": "https://pubs.ext.vt.edu/ENTO/ENTO-453/ENTO-453.html"},
    {"id": "hot_dry", "text": "Do not spray when RH is below 40% and air temperature is above 25 C.",
     "source": "Ontario, How weather conditions affect spray applications", "url": "https://www.ontario.ca/page/how-weather-conditions-affect-spray-applications"},
]
RAIN_NOTE_PCT = 30  # ILLUSTRATIVE: no source gives a number; the label's rainfast period decides


def check_weather(temp_c, humidity_pct, wind_kmh, rain_pct):
    wait = []
    if wind_kmh > 16:
        wait.append(RULES[0])
    if humidity_pct < 40 and temp_c > 25:
        wait.append(RULES[1])
    notes = ["Always read the product label: its weather limits override these rules."]
    if rain_pct >= RAIN_NOTE_PCT:
        notes.append(f"Rain chance {rain_pct}%: check the label's rainfast period before spraying.")
    return {"decision": "WAIT" if wait else "GO (if the label agrees)", "why": wait, "notes": notes}


if __name__ == "__main__":
    print(check_weather(30, 35, 8, 10)["decision"], "|", check_weather(24, 60, 6, 0)["decision"])
