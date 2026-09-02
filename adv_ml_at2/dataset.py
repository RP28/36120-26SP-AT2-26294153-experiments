from datetime import date
from pathlib import Path

from loguru import logger
import pandas as pd
import requests
import typer

from adv_ml_at2.config import (
    MAX_EXPERIMENT_DATE,
    OPEN_METEO_ARCHIVE_URL,
    RAW_DATA_DIR,
    REQUIRED_DAILY_WEATHER_VARIABLES,
    REQUIRED_WEATHER_VARIABLES,
    SYDNEY_LATITUDE,
    SYDNEY_LONGITUDE,
    SYDNEY_TIMEZONE,
)

app = typer.Typer()


def validate_dates(start_date: str, end_date: str) -> None:
    """
    Validate the requested historical data period.

    Parameters
    ----------
    start_date:
        First date to retrieve in YYYY-MM-DD format.
    end_date:
        Final date to retrieve in YYYY-MM-DD format.
    """
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError as exc:
        raise ValueError("Dates must be provided in YYYY-MM-DD format.") from exc
    if start > end:
        raise ValueError("start_date must be before or equal to end_date.")
    if end > MAX_EXPERIMENT_DATE:
        raise ValueError("Data from 2026 onwards cannot be used during experimentation.")

def fetch_historical_weather(
    start_date: str,
    end_date: str,
    additional_hourly_variables: list[str] | None = None,
    daily_variables: list[str] | None = None
) -> dict:
    """
    Fetch historical weather observations for Sydney.

    The hourly variables required to construct CCI and WHC are always
    retrieved. Additional hourly and daily Open-Meteo variables can
    optionally be requested for experimentation.

    Parameters
    ----------
    start_date:
        First date to retrieve in YYYY-MM-DD format.
    end_date:
        Final date to retrieve in YYYY-MM-DD format.
    additional_hourly_variables:
        Optional additional Open-Meteo hourly variables to retrieve.
    daily_variables:
        Optional Open-Meteo daily variables to retrieve.

    Returns
    -------
    dict
        Raw JSON response returned by the Open-Meteo API.
    """
    validate_dates(start_date, end_date)
    additional_hourly_variables = additional_hourly_variables or []
    daily_variables = daily_variables or []
    hourly_variables = list(
        dict.fromkeys(REQUIRED_WEATHER_VARIABLES + additional_hourly_variables))
    daily_variables = list(dict.fromkeys(daily_variables))
    params = {
        "latitude": SYDNEY_LATITUDE,
        "longitude": SYDNEY_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_variables),
        "timezone": SYDNEY_TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    if daily_variables:
        params["daily"] = ",".join(daily_variables)
    logger.info(f"Fetching historical weather data from {start_date} to {end_date}...")
    logger.info(f"Requesting {len(hourly_variables)} hourly variables: {', '.join(hourly_variables)}")
    if daily_variables:
        logger.info(f"Requesting {len(daily_variables)} daily variables: {', '.join(daily_variables)}")
    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Open-Meteo request failed: {exc}") from exc
    data = response.json()
    if "hourly" not in data:
        raise ValueError("Open-Meteo response does not contain hourly weather data.")
    if daily_variables and "daily" not in data:
        raise ValueError("Daily weather variables were requested, but the response does not contain daily data.")
    logger.success("Historical weather data retrieved successfully.")
    return data

def hourly_weather_to_dataframe(data: dict) -> pd.DataFrame:
    """
    Convert hourly Open-Meteo weather data to tabular format.

    Parameters
    ----------
    data:
        Raw Open-Meteo JSON response.

    Returns
    -------
    pd.DataFrame
        Hourly weather observations with one row per timestamp.
    """
    hourly_data = data["hourly"]
    df = pd.DataFrame(hourly_data)
    if "time" not in df.columns:
        raise ValueError("Hourly weather response does not contain a time field.")
    df["time"] = pd.to_datetime(df["time"])
    return df

def daily_weather_to_dataframe(data: dict) -> pd.DataFrame:
    """
    Convert daily Open-Meteo weather data to tabular format.

    Parameters
    ----------
    data:
        Raw Open-Meteo JSON response containing daily observations.

    Returns
    -------
    pd.DataFrame
        Daily weather observations with one row per date.
    """
    if "daily" not in data:
        raise ValueError("Open-Meteo response does not contain daily weather data.")
    daily_data = data["daily"]
    df = pd.DataFrame(daily_data)
    if "time" not in df.columns:
        raise ValueError("Daily weather response does not contain a time field.")
    df["time"] = pd.to_datetime(df["time"])
    return df

def collect_historical_weather(
    start_date: str,
    end_date: str,
    additional_hourly_variables: list[str] | None = None,
    daily_variables: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect historical hourly and daily Sydney weather observations.

    Parameters
    ----------
    start_date:
        First date to retrieve.
    end_date:
        Final date to retrieve.
    additional_hourly_variables:
        Optional additional Open-Meteo hourly variables.
    daily_variables:
        Optional Open-Meteo daily variables. Defaults to required daily variables.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Historical hourly and daily weather observations.
    """
    daily_variables = daily_variables or REQUIRED_DAILY_WEATHER_VARIABLES
    data = fetch_historical_weather(start_date=start_date, end_date=end_date,
        additional_hourly_variables=additional_hourly_variables, daily_variables=daily_variables)
    return hourly_weather_to_dataframe(data), daily_weather_to_dataframe(data)

@app.command()
def main(
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    hourly_output_path: Path = RAW_DATA_DIR / "sydney_weather_hourly.csv",
    daily_output_path: Path = RAW_DATA_DIR / "sydney_weather_daily.csv",
    additional_hourly_variables: list[str] | None = None,
    daily_variables: list[str] | None = None
):
    """
    Download the required historical Sydney weather observations.
    """
    hourly_weather_df, daily_weather_df = collect_historical_weather(start_date=start_date, end_date=end_date,
                                                                     additional_hourly_variables=additional_hourly_variables, 
                                                                     daily_variables=daily_variables)
    hourly_output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_weather_df.to_csv(hourly_output_path, index=False)
    daily_weather_df.to_csv(daily_output_path, index=False)
    logger.info(f"Collected {len(hourly_weather_df):,} hourly observations.")
    logger.info(f"Hourly dataset contains {hourly_weather_df.shape[1]} columns.")
    logger.success(f"Raw hourly weather data saved to: {hourly_output_path}")
    logger.info(f"Collected {len(daily_weather_df):,} daily observations.")
    logger.info(f"Daily dataset contains {daily_weather_df.shape[1]} columns.")
    logger.success(f"Raw daily weather data saved to: {daily_output_path}")


if __name__ == "__main__":
    app()
