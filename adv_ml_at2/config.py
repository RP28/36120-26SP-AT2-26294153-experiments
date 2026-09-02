from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Open-Meteo configuration
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

SYDNEY_LATITUDE = -33.8688
SYDNEY_LONGITUDE = 151.2093
SYDNEY_TIMEZONE = "Australia/Sydney"

# Data from 2026 onwards must not be used during experimentation.
MAX_EXPERIMENT_DATE = date(2025, 12, 31)

# Variables required to construct CCI and WHC.
REQUIRED_WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "precipitation",
    "snowfall",
]

# Daily variables matching the required hourly weather families.
DEFAULT_DAILY_WEATHER_VARIABLES = [
    "weather_code",
    "wind_direction_10m_dominant",
    "daylight_duration",
]


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
