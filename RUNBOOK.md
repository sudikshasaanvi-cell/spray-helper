# Smart Spray Helper - Build Runbook for opencode

Give this file to opencode. It covers the environment, the Git repo, dataset download, training, the field test, the advice search, the app, deployment and the results files. Where a human must act (logins, tokens, clicks), the step is marked STOP-FOR-HUMAN and the agent must stop and ask.


## 0. How to use this runbook

- Make an empty folder. Unzip spray_helper_code.zip into it so severity.py, field_sim.py, weather_rules.py, app.py and requirements.txt are present. Put this runbook and AGENTS.md in the same folder.
- Open a terminal in that folder and start opencode.
- Paste this first message (if the PDF cannot be read, use RUNBOOK.md, which has the same content):

```
Read RUNBOOK.pdf (or RUNBOOK.md) and AGENTS.md. Work phase by phase.
At every STOP-FOR-HUMAN step, stop and tell me exactly what to do.
After each phase run its CHECK, print the result, and git commit.
If a step fails twice, stop and report the error instead of improvising.
```

| Phase | What | Time |
|---|---|---|
| 0 | Environment | 20 min |
| 1 | Git repo | 10 min |
| 2 | Download datasets | 20 min |
| 3 | Train the model | 60-90 min |
| 4 | Predict + Grad-CAM + spray grid | 30 min |
| 5 | Field test | 20 min |
| 6 | Severity + virtual field | 30 min |
| 7 | Advice search (hybrid) | 60 min |
| 8 | Streamlit app | 45 min |
| 9 | Deployment (optional) | 45 min cap |
| 10 | Results + README | 20 min |


## 1. Rules for the agent (non-negotiable)

- Honesty: never invent numbers. Every accuracy or percentage must come from a program output saved in results/.
- Never invent pesticide names, doses or advice. Advice text comes only from docs/ files and is shown with its source URL.
- Secrets: never print, commit or ask for passwords or tokens in chat. kaggle.json and tokens stay outside the repo.
- Stay inside the project folder and its virtual environment. No sudo. No deleting anything outside the folder.
- One module at a time: write it, run it, fix it, commit it.
- Do not rewrite severity.py, field_sim.py or weather_rules.py. Fix a real bug only if you find one, and say so.
- Every assumption (thresholds, litres, seconds) is a named constant with a comment saying it is an assumption.
- If the same step fails twice, stop and report the error and what you tried.
- Keep code simple, with short plain-English comments a student can explain.


## 2. Phase 0 - Environment (20 min)

- Detect the operating system. Check python --version. Use Python 3.10 to 3.12 (TensorFlow may not support the newest Python). If the version is outside this range, STOP-FOR-HUMAN to install Python 3.11.
- Create and activate a virtual environment: python -m venv .venv
- Replace requirements.txt with the list below, then pip install -r requirements.txt:

```
tensorflow
numpy
opencv-python
matplotlib
seaborn
scikit-learn
pillow
streamlit
qdrant-client[fastembed]
kaggle
requests
beautifulsoup4
pypdf
```

- Check for a GPU: python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
- Create config.py with named constants: SEED=42, BATCH=32, CONF_MIN=0.6 (assumption, chosen by eye), HEAT_THRESH=0.5 (assumption). If a GPU is found: IMG_SIZE=224, MAX_PER_CLASS=400. CPU only: IMG_SIZE=160, MAX_PER_CLASS=200.

**CHECK:** Print the Python, TensorFlow, OpenCV and Streamlit versions and the GPU list. All imports succeed.


## 3. Phase 1 - Git repo (10 min)

- git init, set the branch to main, create .gitignore containing: .venv/  data/  field_test/  __pycache__/  *.zip  kaggle.json  qdrant_db/  .env  Make the first commit.

> **STOP-FOR-HUMAN: on github.com create an EMPTY repository named spray-helper and give the agent its URL. (Alternative: log in to the GitHub CLI yourself with gh auth login.) Never paste passwords into the chat. Git will open a browser or credential prompt for you.**

- Then: git remote add origin <URL> and git push -u origin main.

**CHECK:** git log --oneline and git remote -v show the commit and the remote.


## 4. Phase 2 - Download the datasets (20 min)

> **STOP-FOR-HUMAN (Kaggle token): sign in to kaggle.com, open your profile Settings or Account page, find the API section and choose Create New API Token. This downloads kaggle.json. Move it to ~/.kaggle/kaggle.json (Windows: C:\Users\<you>\.kaggle\kaggle.json). On Mac/Linux also run chmod 600 ~/.kaggle/kaggle.json. If Kaggle shows a different token style, follow what its page says. Tell the agent when done.**

Training photos (tomato only, 10 disease classes, about 1000 images each per a paper that used it - verify, do not assume):

```
kaggle datasets download -d kaustubhb999/tomatoleaf -p data --unzip
```

- Show the folder tree 3 levels deep. Find the class folders (one per disease). Write class names and image counts to results/dataset_summary.txt.
- Fallback 1: if the command fails, the human downloads the zip from https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf into data/ and the agent unzips it.
- Fallback 2: full PlantVillage https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset - keep only folders whose names start with Tomato.

Field-test photos (real-world, from PlantDoc, CC BY 4.0):

```
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset data/plantdoc
```

- Find the cropped classification images (train and test folders). List the tomato-related class folders.
- Build field_test/ with only the classes that also exist in the training data (early blight, late blight, septoria leaf spot, bacterial spot, leaf mold, mosaic virus, yellow leaf curl virus, healthy). Class names differ between datasets.
- Print the mapping table (PlantDoc folder -> training class) and save it to results/field_mapping.txt.

> **STOP-FOR-HUMAN: confirm the class mapping looks right before using it.**

**CHECK:** results/dataset_summary.txt exists with counts; the mapping is confirmed.


## 5. Phase 3 - Train the model (60-90 min, run in the background)

- Write train.py using config.py. Use the class folders found in Phase 2 (tomato only). Use at most MAX_PER_CLASS images per class. Hold out 20% for validation (stratified, SEED), no overlap with training.
- Model: MobileNetV2 with ImageNet weights and include_top=False, mobilenet_v2.preprocess_input inside the model, frozen base, GlobalAveragePooling2D, Dropout(0.2), Dense(num_classes, softmax). Light augmentation: flip, rotation 0.1, zoom 0.1.
- Train 5 epochs with the base frozen, then unfreeze the last 30 base layers, learning rate 1e-5, train 3 more epochs. Log time per epoch.
- Save: model/tomato.keras, model/class_names.json, results/accuracy.txt (validation accuracy), results/confusion_matrix.png, results/classification_report.txt.
- If one epoch takes more than 10 minutes on CPU, reduce MAX_PER_CLASS or IMG_SIZE and tell the human.
- Optional GPU fallback: STOP-FOR-HUMAN to push the repo, open Google Colab, git clone it, run train.py, and download model/ and results/ back.

**CHECK:** The three files exist and validation accuracy is printed. If accuracy is below 85%, report it, try at most one change (more epochs or images), then continue. Check model/tomato.keras is under 100 MB. Commit model and results.


## 6. Phase 4 - Predict, Grad-CAM, spray grid (30 min)

- predict.py: load() returns the model and class names. predict(img_bgr) returns (class_name, confidence); below CONF_MIN the result is 'retake'.
- gradcam(model, img_bgr, class_index) returns a heatmap (0 to 1, image size) and an overlay. The MobileNetV2 base is a nested layer; use its last conv layer (out_relu). If a layer name is not found, print the layer names.
- grid_decision(heatmap, confidence, rows=3, cols=3) returns a 3x3 list of 'spray' or 'skip' using HEAT_THRESH; all 'retake' if confidence is low.

**CHECK:** Run on 3 validation photos. Save results/gradcam_1.png to gradcam_3.png and print class, confidence and the grid.


## 7. Phase 5 - Field test (20 min)

- field_eval.py: run the model on field_test/. Print accuracy and per-class results. Write results/field_accuracy.txt and results/field_confusion.png.
- Print one line: lab validation accuracy X, field accuracy Y, drop Z points. Report the number of field photos used; if few, say so.

**CHECK:** results/field_accuracy.txt exists.


## 8. Phase 6 - Severity and virtual field (30 min)

- Pick 9 photos from the validation set automatically into samples/: 3 healthy, 4 diseased from different classes, and 2 deliberately blurred copies of diseased photos (cv2.GaussianBlur, kernel 51). Name them 01 to 09 and list which are blurred in results/field_inputs.txt.
- build_field.py: for each photo run predict(); 'retake' -> ('retake', None); class contains 'healthy' -> ('healthy', None); else severity.estimate_severity -> ('infected', level). Fill the 3x3 grid row by row and call field_sim.compare(grid).
- Save results/field_modeA.png, results/field_modeB.png and results/field_summary.txt (litres, seconds, savings). Print the assumption constants used.
- Run severity.py on 5 diseased validation photos. Save overlays to results/severity_1.png to severity_5.png and a summary to results/severity_summary.txt. Note: severity is colour-based and works best on plain-background single-leaf photos.
- Report what actually happened to the blurred photos; do not force them to be 'retake'.

**CHECK:** The three field_* files and the severity files exist.


## 9. Phase 7 - Advice search (60 min)

Collect documents (ONLY from these sources):

- https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html (early blight)
- https://agritech.tnau.ac.in/crop_protection/tomato_diseases_8.html (late blight)
- The other tomato disease pages on agritech.tnau.ac.in/crop_protection, found by following links from those pages.
- https://niphm.gov.in/IPMPackages/Tomato.pdf (NIPHM AESA-based IPM package for tomato)

- Use requests + BeautifulSoup for pages and pypdf for the PDF. Write one .txt per disease into docs/. Line 1 is the source URL. The rest is the page text exactly as written: no summarizing, nothing added. Target 8 to 15 files covering the training classes.
- rag.py: read docs/, split into chunks of about 250 words with 40 words overlap, keep the source with each chunk. Local Qdrant collection with a dense vector (fastembed sentence-transformers/all-MiniLM-L6-v2, 384 dims, cosine) and a BM25 sparse vector (fastembed Qdrant/bm25, IDF modifier).
- search(query, k=5, mode='hybrid'|'dense'|'bm25') returns text, source, score. Hybrid = two prefetches (dense and sparse) merged with Reciprocal Rank Fusion.
- advice_for(disease_name) searches '<disease> symptoms and management' and returns the top 2 chunks WITH sources, as written. It must not generate or paraphrase. Return None if the best score is weak.
- rag_questions.csv: the agent drafts 15 farmer-style questions (different wording from the documents) with the expected source file. eval_rag.py prints the top-5 hit rate for bm25, dense and hybrid to results/rag_eval.txt.

> **STOP-FOR-HUMAN: review and edit the 15 questions. Questions written by the builders tend to inflate the score; say so honestly at the review.**

**CHECK:** Three hit-rate numbers are printed and saved.


## 10. Phase 8 - Streamlit app (45 min)

- Extend app.py and keep its existing tabs. Add a first tab 'Diagnose': upload a photo; show disease and confidence ('retake' -> warning and stop); severity metric with level; Grad-CAM overlay; the 3x3 spray grid; advice from rag.advice_for with the source as a link, or 'Ask your agriculture officer' when it returns None.
- Load the model once with st.cache_resource.
- Test: run streamlit run app.py --server.headless true --server.port 8501 in the background, request http://localhost:8501/_stcore/health, then stop it.

> **STOP-FOR-HUMAN: open the app in a browser, click through every tab, and save two screenshots (Diagnose and Virtual field) into results/screenshots/.**

**CHECK:** The health check passes and screenshots exist.


## 11. Phase 9 - Deployment (optional, 45 minute cap)

The review can run locally. Do this phase only if Phases 0 to 8 are done.

Option A: Streamlit Community Cloud. It launches apps directly from a GitHub repo. The requirements file must be in the repo root or next to the entrypoint, and the app should be developed in the same Python version used to deploy it (chosen under Advanced settings).

- Agent prep: the repo holds app.py, requirements.txt in the root with versions pinned from the working environment (use tensorflow-cpu if the cloud build is slow or fails), model/, docs/, config.py and all .py modules. No data/ folder. The Qdrant index is built in memory at startup. fastembed downloads small models on first run, so the first load is slow. Commit and push.

> **STOP-FOR-HUMAN: go to share.streamlit.io, sign in with GitHub, choose New app, pick the spray-helper repo, branch main, main file app.py, open Advanced settings and set the Python version to match your local one, then Deploy. Paste any build error to the agent. Allow at most 2 fix attempts.**

Option B (fallback): run locally at the review and show a 2-minute screen recording as backup.

**CHECK:** The deployed URL opens and the Diagnose tab works with one sample photo.


## 12. Phase 10 - Results and README (20 min)

- results/RESULTS.md: collect the numbers from the results files into one table: validation accuracy; field accuracy and the drop; severity examples; Mode A vs Mode B (litres, seconds, savings); RAG hit rates for bm25, dense and hybrid; number of docs and questions; the list of assumptions. Mark anything not completed as NOT DONE.
- README.md: what it is, how to install, get data, train, run and deploy; honest limitations; dataset citations (PlantVillage: Hughes and Salathe, arXiv 1511.08060; PlantDoc: Singh et al., arXiv 1911.10317, CC BY 4.0); sources for the weather rules and advice documents.
- Final message to the human: the RESULTS table and three lists - WORKS, PARTIAL, NOT DONE. Then git commit and push.


## 13. What the human must do (summary)

- Install Python 3.10 to 3.12 if missing.
- Create the empty GitHub repo.
- Create the Kaggle token and place kaggle.json.
- Confirm the PlantDoc class mapping.
- Review the 15 search questions.
- Click through the app and take 2 screenshots.
- Streamlit Cloud clicks (optional).
- Copy the numbers from RESULTS.md into slide 9 of the deck.


## 14. Existing code (do not rewrite)

- severity.py: estimate_severity(img_bgr) -> {'pct', 'level', 'overlay'}. Levels: Mild up to 12%, Moderate up to 50%, Severe above (grouping of Horsfall-Barratt classes; our simplification).
- field_sim.py: compare(grid), draw(grid, result, title), simulate(grid, mode). Cell = ('infected', 'Mild'|'Moderate'|'Severe'), ('healthy', None) or ('retake', None). Litres, seconds and dose factors are assumptions.
- weather_rules.py: check_weather(temp_c, humidity_pct, wind_kmh, rain_pct). Rules from US/Canadian extension guidance; the product label always overrides.
- app.py: Streamlit app with three tabs (severity, virtual field, weather).


## 15. Sources

- PlantVillage on Kaggle (tomato subset): https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf
- PlantVillage on Kaggle (full): https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- PlantVillage on GitHub: https://github.com/spMohanty/PlantVillage-Dataset
- PlantDoc (cropped, classification): https://github.com/pratikkayal/PlantDoc-Dataset  Paper: https://arxiv.org/pdf/1911.10317
- Kaggle API credentials and usage: https://github.com/Kaggle/kaggle-api
- Qdrant hybrid search: https://qdrant.tech/documentation/search-tuning/hybrid-search/
- Streamlit Community Cloud, deploy and dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- TNAU Agritech Portal, tomato early blight: https://agritech.tnau.ac.in/crop_protection/tomato_diseases_2.html
- NIPHM tomato IPM package: https://niphm.gov.in/IPMPackages/Tomato.pdf
- Horsfall-Barratt scale: https://en.wikipedia.org/wiki/Horsfall%E2%80%93Barratt_scale
- Spray weather guidance: https://pubs.ext.vt.edu/ENTO/ENTO-453/ENTO-453.html and https://www.ontario.ca/page/how-weather-conditions-affect-spray-applications
