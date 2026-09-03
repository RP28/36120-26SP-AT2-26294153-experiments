from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer

from adv_ml_at2.config import PROCESSED_DATA_DIR

app = typer.Typer()

def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """
    Validate that all required columns are available.

    Parameters
    ----------
    df:
        DataFrame to validate.
    required_columns:
        Columns required for the operation.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def calculate_cci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Climate Comfort Index for each weather observation.

    Parameters
    ----------
    df:
        Weather observations containing the variables required
        to calculate CCI.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with CCI component scores
        and the final CCI.
    """
    required_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "cloud_cover",
        "precipitation",
    ]
    validate_required_columns(df=df, required_columns=required_columns)
    result = df.copy()
    result["temperature_score"] = (1 - (result["temperature_2m"] - 22).abs() / 20).clip(lower=0)
    result["humidity_score"] = (1 - (result["relative_humidity_2m"] - 50).abs() / 50).clip(lower=0)
    result["wind_score"] = (1 - (result["wind_speed_10m"] - 10).abs() / 40).clip(lower=0)
    result["cloud_score"] = (1 - result["cloud_cover"] / 100).clip(lower=0)
    result["rain_score"] = (1 - result["precipitation"] / 20).clip(lower=0)
    result["cci"] = 100 * (
        0.35 * result["temperature_score"]
        + 0.20 * result["humidity_score"]
        + 0.15 * result["wind_score"]
        + 0.15 * result["cloud_score"]
        + 0.15 * result["rain_score"]
    )
    return result

def calculate_whi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Weather Hazard Index for each weather observation.

    Parameters
    ----------
    df:
        Weather observations containing the variables required
        to calculate WHI.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with hazard component scores
        and the final WHI.
    """
    required_columns = [
        "precipitation",
        "wind_gusts_10m",
        "cloud_cover",
        "snowfall",
        "temperature_2m",
    ]
    validate_required_columns(df=df, required_columns=required_columns)
    result = df.copy()
    result["rain_hazard"] = (result["precipitation"] / 30).clip(upper=1)
    result["wind_hazard"] = (result["wind_gusts_10m"] / 100).clip(upper=1)
    result["cloud_hazard"] = (result["cloud_cover"] / 100).clip(upper=1)
    result["snow_hazard"] = (result["snowfall"] / 15).clip(upper=1)
    result["temperature_hazard"] = ((result["temperature_2m"] - 22).abs() / 25).clip(upper=1)
    result["whi"] = 100 * (
        0.30 * result["rain_hazard"]
        + 0.30 * result["wind_hazard"]
        + 0.20 * result["cloud_hazard"]
        + 0.10 * result["snow_hazard"]
        + 0.10 * result["temperature_hazard"]
    )
    return result

def calculate_whc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Weather Hazard Index values into WHC classes.

    Parameters
    ----------
    df:
        DataFrame containing the Weather Hazard Index.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with WHC class and label.
    """
    validate_required_columns(df=df, required_columns=["whi"])
    result = df.copy()
    result["whc"] = pd.cut(result["whi"], bins=[float("-inf"), 25, 50, 75, float("inf")],
        labels=[0, 1, 2, 3], right=False).astype("int64")
    whc_labels = {
        0: "Low Risk",
        1: "Moderate Risk",
        2: "High Risk",
        3: "Extreme Risk",
    }
    result["whc_label"] = result["whc"].map(whc_labels)
    return result

def aggregate_daily_cci(
    hourly_cci_df: pd.DataFrame,
    time_column: str = "time",
) -> pd.DataFrame:
    """
    Aggregate hourly CCI values into one daily CCI value.

    Daily CCI is represented by the mean of the hourly CCI values
    available for each calendar day.

    Parameters
    ----------
    hourly_cci_df:
        Hourly weather observations containing a 'cci' column.
    time_column:
        Name of the datetime column.

    Returns
    -------
    pd.DataFrame
        Daily CCI values with one row per calendar day.
    """
    validate_required_columns(df=hourly_cci_df, required_columns=[time_column, "cci"])
    result = hourly_cci_df[[time_column, "cci"]].copy()
    result[time_column] = pd.to_datetime(result[time_column])
    result["date"] = result[time_column].dt.normalize()
    daily_cci_df = (
        result.groupby("date", as_index=False)
        .agg(cci=("cci", "mean"))
        .rename(columns={"date": time_column})
    )
    return daily_cci_df

def create_cci_forecast_targets(
    daily_weather_df: pd.DataFrame,
    time_column: str = "time",
    horizons: list[int] = [1, 2, 3]
) -> pd.DataFrame:
    """
    Create CCI regression targets for one, two and three calendar days ahead.

    Target values are matched by calendar date rather than by row position.
    This prevents a missing calendar day from incorrectly changing the
    intended forecast horizon.

    Parameters
    ----------
    daily_weather_df:
        Daily dataset containing the current-day 'cci' value.
    time_column:
        Name of the date column.
    horizons:
        List of forecast horizons in calendar days.

    Returns
    -------
    pd.DataFrame
        Copy of the daily dataset with CCI targets for the specified horizons.
    """
    validate_required_columns(df=daily_weather_df, required_columns=[time_column, "cci"])
    result = daily_weather_df.copy()
    result[time_column] = pd.to_datetime(result[time_column]).dt.normalize()
    result = result.sort_values(time_column).reset_index(drop=True)
    if result[time_column].duplicated().any():
        raise ValueError(f"Daily dataset must contain only one observation per '{time_column}'.")
    cci_by_date = result.set_index(time_column)["cci"]
    for horizon in horizons:
        target_dates = result[time_column] + pd.Timedelta(days=horizon)
        result[f"cci_target_{horizon}d"] = target_dates.map(cci_by_date)
    return result

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
