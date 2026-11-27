# Smart Spray Helper (tomato)

Tomato-leaf photo tool: shows the disease, how bad it is (% leaf area),
which parts of a field to spray, pesticide saved vs spraying everything,
and official advice with its source.

## Install
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
```
kaggle datasets download -d kaustubhb999/tomatoleaf -p data --unzip
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset data/plantdoc
```

## Train
```
python train.py
```

## Run
```
streamlit run app.py
```

## Deploy
Optional: Streamlit Community Cloud. Not done in this build.

## Limitations
- Severity is colour-based; works best on plain-background single-leaf photos.
- Field accuracy much lower than lab (domain shift).
- Weather rules from US/Canadian extension guidance; product label overrides.
- Advice only from docs/ files; no match -> "Ask your agriculture officer."

## Dataset citations
- PlantVillage: Hughes and Salathe, arXiv 1511.08060
- PlantDoc: Singh et al., arXiv 1911.10317, CC BY 4.0

## Sources
- TNAU Agritech Portal tomato disease pages
- NIPHM tomato IPM package: https://niphm.gov.in/IPMPackages/Tomato.pdf
- Spray weather guidance: VT ENTO-453, Ontario spray applications page
