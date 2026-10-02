"""
Run parallel UNSEEN GEV resampling using daily bias-corrected IFS data.

Daily bias correction:
For each region, each bias-correction period, and each IFS day position
(day 1 ... day 30), calculate

    daily_bias(day) =
        ERA5 mean for that day position across years
        - IFS ensemble mean for that day position across years and members

using finite values only.

ERA5 contains up to 31 calendar days whereas IFS contains 30 day positions.
Only the 30 day positions that exist in IFS are matched. ERA5 day 31 is
therefore not used for bias correction because there is no corresponding
IFS value.

NaNs are handled independently for every region / period / day. If one
matched day position has no finite ERA5 or IFS values at all, the code
falls back to a period-mean additive correction calculated from all finite
values in the 30 mutually matchable day positions. This prevents a single
missing calendar day from leaving an IFS day uncorrected.

After bias correction, the original UNSEEN bootstrap procedure is unchanged:
for each region and repetition, one ensemble member is independently
sampled for each year. The selected daily values over 1981-2024 are
concatenated and fitted with a GEV distribution.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import warnings

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.stats import genextreme
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

IFS_INPUT_FILE = Path(
    "IFS_HottestMonth_Daily_Regional_1981_2024.nc"
)

ERA5_INPUT_FILE = Path(
    "ERA5_daily_hottest_month_regionmean_1981_2024.nc"
)

# IMPORTANT:
# Use a new output directory so that old uncorrected checkpoint files
# cannot be mixed with the new bias-corrected calculation.
OUTPUT_DIRECTORY = Path(
    "IFS_UNSEEN_GEV_Fidelity_DailyBiasCorrected"
)

START_YEAR = 1981
END_YEAR = 2024

N_BOOTSTRAP = 10_000
RANDOM_SEED = 20260924
COMPRESSION_LEVEL = 4

# Number of independent region workers.
# Keep this <= CPUs allocated by SLURM/JASMIN.
N_WORKERS = 16

# False allows continuation ONLY from the new bias-corrected directory.
# True deletes existing bias-corrected progress and starts again.
RESTART_GEV = False

# Test one region with REGION_STOP = 1.
# Use None for all 237 regions.
REGION_START = 0
REGION_STOP = None

PAUSE_SECONDS_PER_REGION = 0

# scipy.stats.genextreme uses c = -xi.
SHAPE_CONVENTION = "xi"  # "xi" or "scipy_c"


# ============================================================
# Bias-correction periods
# ============================================================

BIAS_PERIODS = [
    (1981, 1990),
    (1991, 2000),
    (2001, 2010),
    (2011, 2019),
    (2020, 2024),
]


# ============================================================
# Output files
# ============================================================

LOCATION_FILE = OUTPUT_DIRECTORY / "IFS_UNSEEN_GEV_location.nc"
SCALE_FILE = OUTPUT_DIRECTORY / "IFS_UNSEEN_GEV_scale.nc"
SHAPE_FILE = OUTPUT_DIRECTORY / "IFS_UNSEEN_GEV_shape.nc"


# ============================================================
# Metadata
# ============================================================

def metadata(parameter):

    if parameter == "location":
        return "GEV location parameter", "K"

    if parameter == "scale":
        return "GEV scale parameter", "K"

    if SHAPE_CONVENTION == "xi":
        return (
            "GEV shape parameter xi, equal to negative scipy c",
            "1",
        )

    return "SciPy genextreme shape parameter c", "1"


# ============================================================
# Initialise NetCDF output
# ============================================================

def initialise_file(path, parameter, region_ids):

    """Create one output file unless a resumable file already exists."""

    if path.exists() and not RESTART_GEV:
        return

    if path.exists():
        path.unlink()

    long_name, units = metadata(parameter)

    with Dataset(path, "w", format="NETCDF4") as output:

        output.createDimension(
            "region_id",
            region_ids.size,
        )

        output.createDimension(
            "bootstrap",
            N_BOOTSTRAP,
        )

        region = output.createVariable(
            "region_id",
            "i2",
            ("region_id",),
        )

        bootstrap = output.createVariable(
            "bootstrap",
            "i4",
            ("bootstrap",),
        )

        result = output.createVariable(
            parameter,
            "f4",
            ("region_id", "bootstrap"),
            zlib=True,
            complevel=COMPRESSION_LEVEL,
            shuffle=True,
            chunksizes=(
                1,
                min(1000, N_BOOTSTRAP),
            ),
            fill_value=np.float32(np.nan),
        )

        completed = output.createVariable(
            "completed",
            "i1",
            ("region_id",),
        )

        failed = output.createVariable(
            "failed_fits",
            "i4",
            ("region_id",),
        )

        region[:] = region_ids

        bootstrap[:] = np.arange(
            1,
            N_BOOTSTRAP + 1,
            dtype=np.int32,
        )

        completed[:] = 0
        failed[:] = 0

        result.long_name = long_name
        result.units = units

        completed.long_name = (
            "region calculation completed flag"
        )

        failed.long_name = (
            "number of failed GEV fits"
        )

        output.title = (
            "Bias-corrected UNSEEN resampled "
            "GEV parameter distribution"
        )

        output.source_file = str(IFS_INPUT_FILE)
        output.reference_file = str(ERA5_INPUT_FILE)

        output.number_of_repetitions = N_BOOTSTRAP
        output.random_seed = RANDOM_SEED

        output.shape_convention = SHAPE_CONVENTION

        output.bias_correction = (
            "Additive daily bias correction by region, period, and "
            "IFS day position: ERA5 daily mean minus IFS ensemble "
            "daily mean; finite values only; ERA5 day 31 ignored; "
            "period-mean fallback if a matched day has no finite data"
        )

        output.bias_periods = (
            "1981-1990; 1991-2000; 2001-2010; "
            "2011-2019; 2020-2024"
        )

        output.sample_construction = (
            "After additive mean bias correction, "
            "independently select one ensemble member "
            "for every year, then concatenate its finite "
            "daily values across all years."
        )


# ============================================================
# Validate checkpoint
# ============================================================

def validate_checkpoint(
    output,
    parameter,
    region_ids,
):

    if len(output.dimensions["region_id"]) != region_ids.size:

        raise ValueError(
            f"{parameter} checkpoint has the wrong region size."
        )

    if len(output.dimensions["bootstrap"]) != N_BOOTSTRAP:

        raise ValueError(
            f"{parameter} checkpoint has a different N_BOOTSTRAP."
        )

    if not np.array_equal(
        output.variables["region_id"][:],
        region_ids,
    ):

        raise ValueError(
            f"{parameter} checkpoint has different region IDs."
        )

    if int(output.random_seed) != RANDOM_SEED:

        raise ValueError(
            f"{parameter} checkpoint has a different random seed."
        )

    if str(output.shape_convention) != SHAPE_CONVENTION:

        raise ValueError(
            f"{parameter} checkpoint has another shape convention."
        )


# ============================================================
# Bias correction
# ============================================================

def apply_bias_correction(
    ifs_temperature,
    era5_temperature,
    years,
    region_ids,
):

    """
    Apply period-wise, day-position-wise additive bias correction.

    Parameters
    ----------
    ifs_temperature : ndarray
        Shape:
        (region, day=30, year=44, member=25)

    era5_temperature : ndarray
        Shape:
        (region, year=44, day=31)

    years : ndarray
        1981-2024

    region_ids : ndarray

    Returns
    -------
    corrected : ndarray
        Bias-corrected IFS temperature.

    biases : ndarray
        Shape:
        (region, period, IFS_day)

        correction(region, period, day)
            = ERA5 daily mean - IFS ensemble daily mean

        Means are calculated using finite values only.

    fallback_used : ndarray
        Boolean array with the same shape as biases.
        True where the daily bias could not be calculated because one side
        had no finite values and a period-mean fallback was used instead.

    Notes
    -----
    ERA5 has 31 day slots while IFS has 30. Only the first 30 day positions
    are mutually matchable and are used for the daily correction.
    ERA5 day 31 is ignored because IFS has no corresponding day position.

    If a matched day has no finite data in ERA5 or IFS, the correction for
    that day falls back to a period-mean bias calculated from all finite
    values in the 30 mutually matchable day positions.
    """

    corrected = ifs_temperature.copy()

    n_regions = ifs_temperature.shape[0]
    n_ifs_days = ifs_temperature.shape[1]
    n_era5_days = era5_temperature.shape[2]
    n_periods = len(BIAS_PERIODS)

    if n_era5_days < n_ifs_days:
        raise ValueError(
            "ERA5 has fewer day positions than IFS, so all IFS days "
            "cannot be matched safely. "
            f"IFS days={n_ifs_days}, ERA5 days={n_era5_days}."
        )

    n_match_days = n_ifs_days

    biases = np.full(
        (n_regions, n_periods, n_ifs_days),
        np.nan,
        dtype=np.float32,
    )

    fallback_used = np.zeros(
        (n_regions, n_periods, n_ifs_days),
        dtype=bool,
    )

    print("")
    print("=" * 70)
    print("Applying additive DAILY bias correction")
    print("=" * 70)
    print(
        f"IFS day positions: {n_ifs_days}; "
        f"ERA5 day positions: {n_era5_days}"
    )
    print(
        f"Matching day positions 1-{n_match_days}. "
        f"ERA5 day(s) {n_match_days + 1}-{n_era5_days} "
        "are not used because IFS has no counterpart."
        if n_era5_days > n_match_days
        else f"Matching all {n_match_days} day positions."
    )

    for period_index, (start, end) in enumerate(BIAS_PERIODS):

        year_mask = (
            (years >= start)
            & (years <= end)
        )

        year_indices = np.where(year_mask)[0]

        print(
            f"\nPeriod {start}-{end}: "
            f"{len(year_indices)} years"
        )

        period_fallback_count = 0

        for region_position in range(n_regions):

            # ------------------------------------------------
            # Period-mean fallback, using ONLY day positions
            # that exist in both datasets (1 ... n_match_days).
            #
            # This is used only if a particular day position
            # contains no finite values on one side.
            # ------------------------------------------------

            era5_period_common = era5_temperature[
                region_position,
                year_indices,
                :n_match_days,
            ]

            ifs_period_common = ifs_temperature[
                region_position,
                :n_match_days,
                year_indices,
                :,
            ]

            era5_period_finite = era5_period_common[
                np.isfinite(era5_period_common)
            ]

            ifs_period_finite = ifs_period_common[
                np.isfinite(ifs_period_common)
            ]

            if era5_period_finite.size == 0:

                raise ValueError(
                    f"No finite ERA5 data for region "
                    f"{region_ids[region_position]}, "
                    f"period {start}-{end}, within the "
                    f"{n_match_days} matchable day positions."
                )

            if ifs_period_finite.size == 0:

                raise ValueError(
                    f"No finite IFS data for region "
                    f"{region_ids[region_position]}, "
                    f"period {start}-{end}."
                )

            period_fallback_bias = (
                np.mean(
                    era5_period_finite,
                    dtype=np.float64,
                )
                -
                np.mean(
                    ifs_period_finite,
                    dtype=np.float64,
                )
            )

            # ------------------------------------------------
            # Daily correction
            # ------------------------------------------------

            for day_index in range(n_ifs_days):

                # ERA5:
                # same day position across all years in period.
                era5_daily = era5_temperature[
                    region_position,
                    year_indices,
                    day_index,
                ]

                era5_daily_finite = era5_daily[
                    np.isfinite(era5_daily)
                ]

                # IFS:
                # same day position across all years and members.
                ifs_daily = ifs_temperature[
                    region_position,
                    day_index,
                    year_indices,
                    :,
                ]

                ifs_daily_finite = ifs_daily[
                    np.isfinite(ifs_daily)
                ]

                if (
                    era5_daily_finite.size > 0
                    and ifs_daily_finite.size > 0
                ):

                    era5_daily_mean = np.mean(
                        era5_daily_finite,
                        dtype=np.float64,
                    )

                    ifs_daily_mean = np.mean(
                        ifs_daily_finite,
                        dtype=np.float64,
                    )

                    correction = (
                        era5_daily_mean
                        - ifs_daily_mean
                    )

                else:

                    # A day can be absent from ERA5 because of
                    # calendar-month length, or can be entirely
                    # missing in either dataset.
                    correction = period_fallback_bias

                    fallback_used[
                        region_position,
                        period_index,
                        day_index,
                    ] = True

                    period_fallback_count += 1

                biases[
                    region_position,
                    period_index,
                    day_index,
                ] = np.float32(correction)

                # Add correction to all years in this period and
                # all ensemble members for this IFS day position.
                #
                # NaNs remain NaN after addition, which is desired.
                corrected[
                    region_position,
                    day_index,
                    year_indices,
                    :,
                ] += np.float32(correction)

        period_biases = biases[:, period_index, :]

        print(
            "  Daily-bias statistics "
            "(ERA5 daily mean - IFS daily ensemble mean):"
        )

        print(
            f"    mean = "
            f"{np.nanmean(period_biases):+.3f} K"
        )

        print(
            f"    min  = "
            f"{np.nanmin(period_biases):+.3f} K"
        )

        print(
            f"    max  = "
            f"{np.nanmax(period_biases):+.3f} K"
        )

        print(
            f"    fallback day-cells = "
            f"{period_fallback_count} / "
            f"{n_regions * n_ifs_days}"
        )

    total_fallback = int(np.count_nonzero(fallback_used))

    print("")
    print("=" * 70)
    print("Daily bias correction completed")
    print(
        f"Total fallback day-cells: {total_fallback} / "
        f"{fallback_used.size}"
    )
    print("=" * 70)
    print("")

    return corrected, biases, fallback_used


# ============================================================
# Save bias information
# ============================================================

def save_bias_file(
    biases,
    fallback_used,
    region_ids,
):

    """
    Save daily corrections for every region / period / IFS day position.
    """

    bias_file = (
        OUTPUT_DIRECTORY
        / "IFS_additive_daily_bias_correction.nc"
    )

    period_labels = [
        f"{start}-{end}"
        for start, end in BIAS_PERIODS
    ]

    n_ifs_days = biases.shape[2]

    ds = xr.Dataset(
        data_vars={
            "bias_correction": (
                ("region_id", "period", "day"),
                biases,
            ),
            "fallback_used": (
                ("region_id", "period", "day"),
                fallback_used.astype(np.int8),
            ),
        },
        coords={
            "region_id": region_ids,
            "period": np.arange(
                len(BIAS_PERIODS),
                dtype=np.int16,
            ),
            "day": np.arange(
                1,
                n_ifs_days + 1,
                dtype=np.int16,
            ),
            "period_label": (
                "period",
                period_labels,
            ),
            "period_start_year": (
                "period",
                np.array(
                    [p[0] for p in BIAS_PERIODS],
                    dtype=np.int16,
                ),
            ),
            "period_end_year": (
                "period",
                np.array(
                    [p[1] for p in BIAS_PERIODS],
                    dtype=np.int16,
                ),
            ),
        },
    )

    ds["bias_correction"].attrs[
        "long_name"
    ] = (
        "Additive daily correction applied to IFS "
        "(ERA5 daily mean minus IFS ensemble daily mean)"
    )

    ds["bias_correction"].attrs["units"] = "K"

    ds["fallback_used"].attrs[
        "long_name"
    ] = (
        "1 where period-mean fallback correction was used "
        "because the matched daily mean could not be calculated"
    )

    ds["fallback_used"].attrs["units"] = "1"

    ds.attrs["method"] = (
        "For each region, period, and IFS day position (1-30), "
        "ERA5 mean is calculated across finite years and IFS mean "
        "across finite years and ensemble members. "
        "Correction = ERA5 daily mean - IFS daily ensemble mean. "
        "ERA5 day 31 is ignored because IFS has only 30 day positions. "
        "If either side has no finite values for a matched day, "
        "a period-mean correction based on all finite values in the "
        "30 mutually matchable day positions is used."
    )

    ds.to_netcdf(bias_file)

    print(
        f"Daily bias correction file saved:\n"
        f"  {bias_file}"
    )


# ============================================================
# GEV fitting
# ============================================================

def fit_region(
    region_data,
    region_id,
):

    """Fit all Monte Carlo samples for one region."""

    n_days, n_years, n_members = region_data.shape

    if (
        n_days,
        n_years,
        n_members,
    ) != (
        30,
        44,
        25,
    ):

        raise ValueError(
            f"Region {region_id}: "
            f"unexpected shape {region_data.shape}."
        )

    # Region-specific stream guarantees identical results
    # after resuming.
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                RANDOM_SEED,
                int(region_id),
            ]
        )
    )

    year_indices = np.arange(n_years)

    location = np.full(
        N_BOOTSTRAP,
        np.nan,
        dtype=np.float32,
    )

    scale = np.full(
        N_BOOTSTRAP,
        np.nan,
        dtype=np.float32,
    )

    shape = np.full(
        N_BOOTSTRAP,
        np.nan,
        dtype=np.float32,
    )

    failed_count = 0

    for repetition in range(N_BOOTSTRAP):

        # One ensemble member independently selected
        # for each year.
        selected_members = rng.integers(
            0,
            n_members,
            size=n_years,
        )

        sample = region_data[
            :,
            year_indices,
            selected_members,
        ].reshape(-1)

        # Remove NaNs, including unavailable calendar days.
        sample = sample[
            np.isfinite(sample)
        ]

        if (
            sample.size < 30
            or np.std(sample) == 0
        ):

            failed_count += 1
            continue

        try:

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore",
                    RuntimeWarning,
                )

                scipy_c, loc, fitted_scale = (
                    genextreme.fit(sample)
                )

            if (
                not np.isfinite(scipy_c)
                or not np.isfinite(loc)
                or not np.isfinite(fitted_scale)
                or fitted_scale <= 0
            ):

                failed_count += 1
                continue

            location[repetition] = np.float32(
                loc
            )

            scale[repetition] = np.float32(
                fitted_scale
            )

            shape[repetition] = np.float32(
                -scipy_c
                if SHAPE_CONVENTION == "xi"
                else scipy_c
            )

        except Exception:

            failed_count += 1

    return (
        location,
        scale,
        shape,
        failed_count,
    )


# ============================================================
# Parallel worker
# ============================================================

def fit_region_worker(
    position,
    region_id,
    region_data,
):

    """
    Picklable process worker.
    NetCDF writing remains in the main process.
    """

    (
        location,
        scale,
        shape,
        failed_count,
    ) = fit_region(
        region_data,
        region_id,
    )

    return (
        position,
        region_id,
        location,
        scale,
        shape,
        failed_count,
    )


# ============================================================
# Main
# ============================================================

def main():

    if SHAPE_CONVENTION not in {
        "xi",
        "scipy_c",
    }:

        raise ValueError(
            "SHAPE_CONVENTION must be "
            "'xi' or 'scipy_c'."
        )

    if not IFS_INPUT_FILE.exists():

        raise FileNotFoundError(
            f"IFS input does not exist:\n"
            f"{IFS_INPUT_FILE}"
        )

    if not ERA5_INPUT_FILE.exists():

        raise FileNotFoundError(
            f"ERA5 input does not exist:\n"
            f"{ERA5_INPUT_FILE}"
        )

    if N_WORKERS < 1:

        raise ValueError(
            "N_WORKERS must be at least 1."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Read IFS
    # ========================================================

    print("Reading IFS data...")

    with xr.open_dataset(
        IFS_INPUT_FILE,
        cache=False,
    ) as source:

        temperature = source["t2m"].transpose(
            "region_id",
            "day",
            "year",
            "number",
        )

        region_ids = (
            source["region_id"]
            .values
            .astype(np.int16)
        )

        years = source["year"].values.copy()

        if temperature.shape != (
            237,
            30,
            44,
            25,
        ):

            raise ValueError(
                f"IFS t2m shape is "
                f"{temperature.shape}; "
                f"expected (237, 30, 44, 25)."
            )

        if not np.array_equal(
            years,
            np.arange(
                START_YEAR,
                END_YEAR + 1,
            ),
        ):

            raise ValueError(
                "IFS year coordinate "
                "is not 1981-2024."
            )

        all_temperature = (
            temperature
            .values
            .astype(
                np.float32,
                copy=True,
            )
        )

    # ========================================================
    # Read ERA5
    # ========================================================

    print("Reading ERA5 data...")

    with xr.open_dataset(
        ERA5_INPUT_FILE,
        cache=False,
    ) as era5_source:

        era5_temperature_da = (
            era5_source["t2m"].transpose(
                "region_id",
                "year",
                "day",
            )
        )

        era5_region_ids = (
            era5_source["region_id"]
            .values
            .astype(np.int16)
        )

        era5_years = (
            era5_source["year"]
            .values
            .copy()
        )

        if era5_temperature_da.shape != (
            237,
            44,
            31,
        ):

            raise ValueError(
                f"ERA5 t2m shape is "
                f"{era5_temperature_da.shape}; "
                f"expected (237, 44, 31)."
            )

        if not np.array_equal(
            era5_region_ids,
            region_ids,
        ):

            raise ValueError(
                "ERA5 and IFS region IDs "
                "do not match."
            )

        if not np.array_equal(
            era5_years,
            years,
        ):

            raise ValueError(
                "ERA5 and IFS year coordinates "
                "do not match."
            )

        era5_temperature = (
            era5_temperature_da
            .values
            .astype(
                np.float32,
                copy=False,
            )
        )

    # ========================================================
    # Additive daily bias correction
    # ========================================================

    (
        all_temperature,
        biases,
        fallback_used,
    ) = apply_bias_correction(
        all_temperature,
        era5_temperature,
        years,
        region_ids,
    )

    # Save corrections for later checking.
    save_bias_file(
        biases,
        fallback_used,
        region_ids,
    )

    # Free ERA5 array.
    del era5_temperature

    # ========================================================
    # Initialise GEV output
    # ========================================================

    initialise_file(
        LOCATION_FILE,
        "location",
        region_ids,
    )

    initialise_file(
        SCALE_FILE,
        "scale",
        region_ids,
    )

    initialise_file(
        SHAPE_FILE,
        "shape",
        region_ids,
    )

    # ========================================================
    # Parallel GEV calculation
    # ========================================================

    with (
        Dataset(
            LOCATION_FILE,
            "r+",
        ) as location_output,

        Dataset(
            SCALE_FILE,
            "r+",
        ) as scale_output,

        Dataset(
            SHAPE_FILE,
            "r+",
        ) as shape_output,
    ):

        validate_checkpoint(
            location_output,
            "location",
            region_ids,
        )

        validate_checkpoint(
            scale_output,
            "scale",
            region_ids,
        )

        validate_checkpoint(
            shape_output,
            "shape",
            region_ids,
        )

        completed = (
            location_output
            .variables["completed"][:]
            .astype(bool)

            &

            scale_output
            .variables["completed"][:]
            .astype(bool)

            &

            shape_output
            .variables["completed"][:]
            .astype(bool)
        )

        stop = (
            region_ids.size
            if REGION_STOP is None
            else REGION_STOP
        )

        if (
            REGION_START < 0
            or stop > 237
            or REGION_START >= stop
        ):

            raise ValueError(
                "Invalid REGION_START "
                "or REGION_STOP."
            )

        pending_positions = [
            position
            for position in range(
                REGION_START,
                stop,
            )
            if not completed[position]
        ]

        print("")
        print(
            f"Worker processes: "
            f"{N_WORKERS}"
        )

        print(
            f"Regions still to calculate: "
            f"{len(pending_positions)}"
        )

        if not pending_positions:

            print(
                "All requested regions "
                "are already complete."
            )

        else:

            process_context = (
                mp.get_context("fork")
            )

            with ProcessPoolExecutor(
                max_workers=N_WORKERS,
                mp_context=process_context,
            ) as executor:

                futures = {

                    executor.submit(
                        fit_region_worker,
                        position,
                        int(
                            region_ids[position]
                        ),
                        all_temperature[
                            position
                        ],
                    ): position

                    for position
                    in pending_positions
                }

                progress = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="GEV regions",
                )

                for future in progress:

                    (
                        position,
                        region_id,
                        location,
                        scale,
                        shape,
                        failed_count,
                    ) = future.result()

                    progress.set_postfix_str(
                        f"saved ID={region_id}"
                    )

                    location_output.variables[
                        "location"
                    ][position, :] = location

                    scale_output.variables[
                        "scale"
                    ][position, :] = scale

                    shape_output.variables[
                        "shape"
                    ][position, :] = shape

                    for output in (
                        location_output,
                        scale_output,
                        shape_output,
                    ):

                        output.variables[
                            "failed_fits"
                        ][position] = (
                            failed_count
                        )

                        output.variables[
                            "completed"
                        ][position] = 1

                        output.sync()

                    completed[position] = True

                    if (
                        PAUSE_SECONDS_PER_REGION
                        > 0
                    ):

                        time.sleep(
                            PAUSE_SECONDS_PER_REGION
                        )

    print("")
    print(
        "GEV calculation completed "
        "or checkpointed successfully."
    )

    print(
        f"Location: {LOCATION_FILE}"
    )

    print(
        f"Scale:    {SCALE_FILE}"
    )

    print(
        f"Shape:    {SHAPE_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()

