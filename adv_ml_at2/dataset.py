from collections.abc import Callable
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import time

from loguru import logger
import pandas as pd
import requests
import typer

from adv_ml_at2.config import (
    DEFAULT_DAILY_WEATHER_VARIABLES,
    MAX_EXPERIMENT_DATE,
    OPEN_METEO_ARCHIVE_URL,
    RAW_DATA_DIR,
    REQUIRED_WEATHER_VARIABLES,
    SYDNEY_LATITUDE,
    SYDNEY_LONGITUDE,
    SYDNEY_TIMEZONE,
    CACHE_DIR
)

app = typer.Typer()

def chunk_date_range(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """
    Split a date range into full calendar-year chunks for fetching.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    return [
        (date(year, 1, 1).isoformat(), date(year, 12, 31).isoformat())
        for year in range(start.year, end.year + 1)
    ]

def _calendar_year_bounds(start_date: str, end_date: str) -> tuple[str, str]:
    """
    Return the full calendar-year bounds for a single-year API request.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start.year != end.year:
        raise ValueError("fetch_historical_weather must be called for one calendar year at a time.")
    return date(start.year, 1, 1).isoformat(), date(start.year, 12, 31).isoformat()

def validate_dates(start_date: str, end_date: str) -> None:
    """Validate the requested historical data period."""
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError as exc:
        raise ValueError("Dates must be provided in YYYY-MM-DD format.") from exc
    if start > end:
        raise ValueError("start_date must be before or equal to end_date.")
    if end > MAX_EXPERIMENT_DATE:
        raise ValueError("Data from 2026 onwards cannot be used during experimentation.")

def _build_api_cache_path(
    url: str,
    params: dict,
    cache_dir: Path = CACHE_DIR,
    cache_signature: dict | None = None,
) -> Path:
    """Build a cache filename unique to an API URL and request parameters."""
    cache_signature = cache_signature or {
        "url": url,
        "params": params,
    }
    signature_text = json.dumps(cache_signature, sort_keys=True)
    signature_hash = sha256(signature_text.encode("utf-8")).hexdigest()[:12]
    start_date = cache_signature.get("start_date", params.get("start_date", "request"))
    end_date = cache_signature.get("end_date", params.get("end_date", start_date))
    return Path(cache_dir) / f"{start_date}_{end_date}_{signature_hash}.json"

def _load_cached_response(cache_path: Path) -> dict | None:
    """Load a cached API response if the cache file is valid."""
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Ignoring invalid cache file {cache_path}: {exc}")
        return None
    return data

def _save_cached_response(data: dict, cache_path: Path) -> None:
    """Save a raw API response to the local cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(data, file)

def _weather_response_matches_request(
    data: dict,
    start_date: str,
    end_date: str,
    hourly_variables: list[str],
    daily_variables: list[str],
) -> bool:
    """Check whether cached weather data matches the requested range and variables."""
    days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
    hourly = data.get("hourly", {})
    hourly_times = hourly.get("time", [])
    if any(variable not in hourly for variable in hourly_variables):
        return False
    if len(hourly_times) != days * 24 or hourly_times[0] != f"{start_date}T00:00" or hourly_times[-1] != f"{end_date}T23:00":
        return False

    if daily_variables:
        daily = data.get("daily", {})
        daily_times = daily.get("time", [])
        if any(variable not in daily for variable in daily_variables):
            return False
        if len(daily_times) != days or daily_times[0] != start_date or daily_times[-1] != end_date:
            return False
    return True

def cached_api_request(
    url: str,
    params: dict,
    cache_signature: dict | None = None,
    cached_response_validator: Callable[[dict], bool] | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    cache_dir: Path = CACHE_DIR,
    max_retries: int = 5,
    retry_backoff_seconds: float = 15.0,
    timeout_seconds: float = 60.0,
) -> dict:
    """Request JSON from an API through the shared local cache."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1.")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds must be non-negative.")
    cache_path = _build_api_cache_path(
        url=url,
        params=params,
        cache_dir=Path(cache_dir),
        cache_signature=cache_signature,
    )
    if use_cache and not force_refresh:
        data = _load_cached_response(cache_path)
        if data is not None and (cached_response_validator is None or cached_response_validator(data)):
            logger.info(f"Loaded cached API response: {cache_path}")
            return data
        if data is not None:
            logger.warning(f"Ignoring cached API response with mismatched data: {cache_path}")
    for attempt in range(max_retries):
        try:
            logger.info(f"Requesting API response: {url}")
            response = requests.get(url, params=params, timeout=timeout_seconds)
        except requests.RequestException as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = float(retry_after) if retry_after else None
            except ValueError:
                wait_seconds = None
            if wait_seconds is None:
                wait_seconds = retry_backoff_seconds * (2 ** attempt)
            if attempt == max_retries - 1:
                break
            logger.warning(f"API rate limit reached. Retrying in {wait_seconds:.0f} seconds...")
            time.sleep(wait_seconds)
            continue
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc
        data = response.json()
        if use_cache:
            _save_cached_response(data, cache_path)
            logger.info(f"Cached API response: {cache_path}")
        return data
    raise RuntimeError(f"API rate limit remained active after {max_retries} attempts.")

def fetch_historical_weather(
    start_date: str,
    end_date: str,
    additional_hourly_variables: list[str] | None = None,
    daily_variables: list[str] | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    cache_dir: Path = CACHE_DIR,
    max_retries: int = 5,
    retry_backoff_seconds: float = 15.0,
) -> dict:
    """Fetch one full calendar year of historical Sydney weather."""
    validate_dates(start_date, end_date)
    start_date, end_date = _calendar_year_bounds(start_date, end_date)
    additional_hourly_variables = additional_hourly_variables or []
    daily_variables = daily_variables or []
    hourly_variables = list(dict.fromkeys(REQUIRED_WEATHER_VARIABLES + additional_hourly_variables))
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
    logger.info(f"Preparing historical weather data from {start_date} to {end_date}...")
    logger.info(f"Hourly weather variables: {', '.join(hourly_variables)}")
    if daily_variables:
        logger.info(f"Daily weather variables: {', '.join(daily_variables)}")
    cache_signature = {
        "latitude": SYDNEY_LATITUDE,
        "longitude": SYDNEY_LONGITUDE,
        "timezone": SYDNEY_TIMEZONE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_variables,
        "daily": daily_variables,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    data = cached_api_request(
        url=OPEN_METEO_ARCHIVE_URL,
        params=params,
        cache_signature=cache_signature,
        cached_response_validator=lambda data: _weather_response_matches_request(
            data=data,
            start_date=start_date,
            end_date=end_date,
            hourly_variables=hourly_variables,
            daily_variables=daily_variables,
        ),
        use_cache=use_cache,
        force_refresh=force_refresh,
        cache_dir=cache_dir,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    if "hourly" not in data:
        raise ValueError("Open-Meteo response does not contain hourly weather data.")
    if daily_variables and "daily" not in data:
        raise ValueError("Daily weather variables were requested, but the response does not contain daily data.")
    logger.success("Historical weather data is available.")
    return data

def hourly_weather_to_dataframe(data: dict) -> pd.DataFrame:
    """Convert hourly Open-Meteo weather data to tabular format."""
    hourly_data = data["hourly"]
    df = pd.DataFrame(hourly_data)
    if "time" not in df.columns:
        raise ValueError("Hourly weather response does not contain a time field.")
    df["time"] = pd.to_datetime(df["time"])
    return df

def daily_weather_to_dataframe(data: dict) -> pd.DataFrame:
    """Convert daily Open-Meteo weather data to tabular format."""
    if "daily" not in data:
        raise ValueError("Open-Meteo response does not contain daily weather data.")
    daily_data = data["daily"]
    df = pd.DataFrame(daily_data)
    if "time" not in df.columns:
        raise ValueError("Daily weather response does not contain a time field.")
    df["time"] = pd.to_datetime(df["time"])
    return df

def _filter_dataframes_to_date_range(
    hourly_weather_df: pd.DataFrame,
    daily_weather_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter full-year weather fetches back to the requested date range."""
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    hourly_weather_df = hourly_weather_df[
        hourly_weather_df["time"].ge(start_timestamp) & hourly_weather_df["time"].lt(end_timestamp)
    ].reset_index(drop=True)
    daily_weather_df = daily_weather_df[
        daily_weather_df["time"].ge(start_timestamp) & daily_weather_df["time"].lt(end_timestamp)
    ].reset_index(drop=True)
    return hourly_weather_df, daily_weather_df

def collect_historical_weather(
    start_date: str,
    end_date: str,
    additional_hourly_variables: list[str] | None = None,
    daily_variables: list[str] | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    cache_dir: Path = CACHE_DIR,
    max_retries: int = 5,
    retry_backoff_seconds: float = 15.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect historical hourly and daily Sydney weather observations."""
    validate_dates(start_date, end_date)
    daily_variables = daily_variables or DEFAULT_DAILY_WEATHER_VARIABLES
    date_chunks = chunk_date_range(
        start_date=start_date,
        end_date=end_date,
    )
    logger.info(f"Processing {len(date_chunks)} calendar-year chunks with cache {'enabled' if use_cache else 'disabled'}...")
    results: list[dict] = []
    for index, (chunk_start, chunk_end) in enumerate(date_chunks, start=1):
        data = fetch_historical_weather(
            start_date=chunk_start,
            end_date=chunk_end,
            additional_hourly_variables=additional_hourly_variables,
            daily_variables=daily_variables,
            use_cache=use_cache,
            force_refresh=force_refresh,
            cache_dir=cache_dir,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        logger.info(f"Collected chunk {index}/{len(date_chunks)}: {chunk_start} to {chunk_end}")
        results.append(data)
    hourly_weather_df = pd.concat([hourly_weather_to_dataframe(data) for data in results], ignore_index=True)
    daily_weather_df = pd.concat([daily_weather_to_dataframe(data) for data in results], ignore_index=True)
    hourly_weather_df = (
        hourly_weather_df
        .drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    daily_weather_df = (
        daily_weather_df
        .drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    return _filter_dataframes_to_date_range(
        hourly_weather_df=hourly_weather_df,
        daily_weather_df=daily_weather_df,
        start_date=start_date,
        end_date=end_date,
    )

def save_weather_dataframes(
    hourly_weather_df: pd.DataFrame,
    daily_weather_df: pd.DataFrame,
    hourly_output_path: Path = RAW_DATA_DIR / "sydney_weather_hourly.csv",
    daily_output_path: Path = RAW_DATA_DIR / "sydney_weather_daily.csv"
) -> tuple[Path, Path]:
    """Save hourly and daily weather observations to CSV files."""
    hourly_output_path = Path(hourly_output_path)
    daily_output_path = Path(daily_output_path)
    hourly_output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_weather_df.to_csv(hourly_output_path, index=False)
    daily_weather_df.to_csv(daily_output_path, index=False)
    return hourly_output_path, daily_output_path

def aggregate_hourly_weather(
    hourly_weather_df: pd.DataFrame,
    aggregation_rules: dict[str, list[str] | str],
    time_column: str = "time",
) -> pd.DataFrame:
    """Aggregate hourly weather observations to daily observations."""
    if time_column not in hourly_weather_df.columns:
        raise ValueError(f"Hourly dataset does not contain '{time_column}'.")
    missing_columns = [column for column in aggregation_rules if column not in hourly_weather_df.columns]
    if missing_columns:
        raise ValueError(f"Aggregation rules contain columns that are not present in the hourly dataset: {missing_columns}")
    df = hourly_weather_df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df["date"] = df[time_column].dt.normalize()
    daily_df = df.groupby("date").agg(aggregation_rules).reset_index()
    daily_df.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in daily_df.columns
    ]
    daily_df = daily_df.rename(columns={"date": "time"})
    return daily_df

def merge_daily_weather(
    aggregated_hourly_df: pd.DataFrame,
    daily_weather_df: pd.DataFrame,
    time_column: str = "time",
) -> pd.DataFrame:
    """Merge hourly-derived daily weather statistics with Open-Meteo daily data."""
    if time_column not in aggregated_hourly_df.columns:
        raise ValueError(f"Aggregated hourly dataset does not contain '{time_column}'.")
    if time_column not in daily_weather_df.columns:
        raise ValueError(f"Daily dataset does not contain '{time_column}'.")
    hourly_daily = aggregated_hourly_df.copy()
    daily = daily_weather_df.copy()
    hourly_daily[time_column] = pd.to_datetime(hourly_daily[time_column]).dt.normalize()
    daily[time_column] = pd.to_datetime(daily[time_column]).dt.normalize()
    merged_df = hourly_daily.merge(daily, on=time_column, how="inner",validate="one_to_one")
    return merged_df

def build_daily_weather_dataset(
    hourly_weather_df: pd.DataFrame,
    daily_weather_df: pd.DataFrame,
    aggregation_rules: dict[str, list[str] | str],
) -> pd.DataFrame:
    """Build a single daily weather dataset from hourly and daily observations."""
    aggregated_hourly_df = aggregate_hourly_weather(hourly_weather_df=hourly_weather_df, aggregation_rules=aggregation_rules)
    return merge_daily_weather(aggregated_hourly_df=aggregated_hourly_df, daily_weather_df=daily_weather_df)

@app.command()
def main(
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    hourly_output_path: Path = RAW_DATA_DIR / "sydney_weather_hourly.csv",
    daily_output_path: Path = RAW_DATA_DIR / "sydney_weather_daily.csv",
    additional_hourly_variables: list[str] | None = None,
    daily_variables: list[str] | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    max_retries: int = 5,
    retry_backoff_seconds: float = 15.0,
):
    """Download the required historical Sydney weather observations."""
    hourly_weather_df, daily_weather_df = collect_historical_weather(
        start_date=start_date,
        end_date=end_date,
        additional_hourly_variables=additional_hourly_variables,
        daily_variables=daily_variables,
        use_cache=use_cache,
        force_refresh=force_refresh,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    hourly_output_path, daily_output_path = save_weather_dataframes(
        hourly_weather_df=hourly_weather_df,
        daily_weather_df=daily_weather_df,
        hourly_output_path=hourly_output_path,
        daily_output_path=daily_output_path,
    )
    logger.info(f"Collected {len(hourly_weather_df):,} hourly observations.")
    logger.info(f"Hourly dataset contains {hourly_weather_df.shape[1]} columns.")
    logger.success(f"Raw hourly weather data saved to: {hourly_output_path}")
    logger.info(f"Collected {len(daily_weather_df):,} daily observations.")
    logger.info(f"Daily dataset contains {daily_weather_df.shape[1]} columns.")
    logger.success(f"Raw daily weather data saved to: {daily_output_path}")


if __name__ == "__main__":
    app()
