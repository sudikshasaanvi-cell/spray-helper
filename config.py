# Named constants (assumptions called out in comments)
SEED = 42
BATCH = 32
CONF_MIN = 0.6          # assumption: chosen by eye, below this we ask to retake the photo
HEAT_THRESH = 0.5      # assumption: heatmap cutoff for spray/skip cells

# CPU only (no GPU detected) -> smaller images and fewer per class
IMG_SIZE = 160
MAX_PER_CLASS = 1000
