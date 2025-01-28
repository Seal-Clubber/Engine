import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

def lomb_scargle_multi_interpolate_auto_k(
    t, y, t_interp,
    n_freq=2000,
    max_k=30,        # maximum number of frequencies to consider
    criterion='BIC', # or 'AIC'
    f_min=None,
    f_max=None,
    normalization='standard'
):
    """
    Interpolate missing data using a multi-frequency Lomb–Scargle fit
    with automatic selection of how many frequencies to include (up to max_k).

    Parameters
    ----------
    t : array-like
        Original time points (non-uniform allowed).
    y : array-like
        Observed values at corresponding times t.
    t_interp : array-like
        Times at which to reconstruct/interpolate.
    n_freq : int, optional
        Number of frequencies in the initial Lomb–Scargle scan.
    max_k : int, optional
        Maximum number of top frequencies to consider.
    criterion : {'BIC', 'AIC'}, optional
        Information criterion to decide how many frequencies to keep.
    f_min, f_max : float, optional
        Frequency range [f_min, f_max]. If None, inferred heuristically.
    normalization : {'standard', 'model', 'psd'}, optional
        LombScargle normalization mode.

    Returns
    -------
    y_interp : ndarray
        Reconstructed/interpolated signal at t_interp.
    k_opt : int
        Number of frequencies chosen.
    freqs_opt : ndarray
        Frequencies used in the final model.
    amps_opt : ndarray
        The fitted amplitude coefficients (size 2*k_opt).
    """

    t = np.asarray(t)
    y = np.asarray(y)
    t_interp = np.asarray(t_interp)

    if len(t) != len(y):
        raise ValueError("Time array t and data array y must have the same length.")

    if len(t) < 2:
        raise ValueError("Need at least 2 data points.")

    # Heuristic frequency range
    t_min, t_max = np.min(t), np.max(t)
    baseline = t_max - t_min
    if f_min is None:
        f_min = 1.0 / (baseline * 10.0)
    if f_max is None:
        dt_min = np.min(np.diff(np.sort(t)))
        f_max = 1.0 / (dt_min * 2.0)

    # Step 1: Lomb–Scargle scan
    freq_grid = np.linspace(f_min, f_max, n_freq)
    ls = LombScargle(t, y, normalization=normalization)
    power = ls.power(freq_grid)

    # Sort frequencies by descending power
    idx_sorted = np.argsort(power)[::-1]
    freq_sorted = freq_grid[idx_sorted]

    # Design matrix builder
    def design_matrix(times, freqs):
        times = times[:, None]  # (n_samples, 1)
        freqs = freqs[None, :]  # (1, k)
        cpart = np.cos(2.0 * np.pi * freqs * times)  # (n_samples, k)
        spart = np.sin(2.0 * np.pi * freqs * times)  # (n_samples, k)
        return np.hstack([cpart, spart])  # (n_samples, 2*k)

    # We'll track the best (lowest) IC value
    best_ic = np.inf
    best_k = 1
    best_amps = None

    # For each k = 1..max_k, build model, fit, compute residual, BIC/AIC
    for k in range(1, min(max_k, len(freq_sorted)) + 1):
        freqs_k = freq_sorted[:k]
        X = design_matrix(t, freqs_k)

        # Fit amplitude coefficients
        amps, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Residual sum of squares
        residuals = y - X @ amps
        rss = np.sum(residuals**2)
        n = len(y)        # number of observations
        p = 2 * k         # number of parameters (cos & sin for each freq)

        # Log-likelihood ~ -n/2 * log(RSS/n)
        # We'll ignore constant factors that don't affect model selection
        ll = -0.5 * n * np.log(rss / n)

        if criterion.upper() == 'AIC':
            # AIC = -2 * ll + 2 * p
            ic_value = -2.0 * ll + 2.0 * p
        else:
            # BIC = -2 * ll + p * ln(n)
            ic_value = -2.0 * ll + p * np.log(n)

        if ic_value < best_ic:
            best_ic = ic_value
            best_k = k
            best_amps = amps

    # Now build final model with best_k frequencies
    freqs_opt = freq_sorted[:best_k]
    # Evaluate at t_interp
    X_new = design_matrix(t_interp, freqs_opt)
    y_interp = X_new @ best_amps

    return y_interp, best_k, freqs_opt, best_amps

def visualizeToPng(x_values, y_values, annotations=None, output_path='./visualize.png'):
    import os
    import matplotlib.pyplot as plt
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.scatter(x_values, y_values, color='blue', label='Data Points')
    # Add annotations if provided
    if annotations:
        for i, annotation in enumerate(annotations):
            plt.text(x_values[i], y_values[i], str(annotation), fontsize=12, ha='center', va='center')
    # Formatting the plot
    plt.title("Time Series Visualization", fontsize=16)
    plt.xlabel("X Values", fontsize=14)
    plt.ylabel("Y Values", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # Rotate x-axis labels for better readability if needed
    plt.xticks(rotation=45)
    # Save the plot to a file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    # Close the plot to free memory
    plt.close()

def combine_original_and_interpolated(t_missing, y_missing, t_interp, y_filled):
    """Combine original and interpolated data, filling gaps with interpolated values."""
    combined_y = np.array(y_filled)  # Start with interpolated data
    # Find indices where t_interp matches t_missing
    indices_missing = np.isin(t_interp, t_missing)
    combined_y[indices_missing] = y_missing  # Replace with original data where available
    return t_interp, combined_y


def lomb_scargle_search(
    t, y, t_interp,
    n_freq=2048,
    max_k=32,        # maximum number of frequencies to consider
    criterion='BIC', # or 'AIC'
    f_min=None,
    f_max=None,
    normalization='standard'
):
    from satoriengine.veda.interpolation.middle_out import middle_out
    max_k_pattern = middle_out(max_k, bias=True)
    n_freq_pattern = middle_out(n_freq, bias=True)
    max_y = max(y)
    min_y = min(y)
    i = len(n_freq_pattern) * len(max_k_pattern)
    j = 0
    k = 0
    fail_early = min(len(n_freq_pattern), len(max_k_pattern))
    while i > 0 and fail_early > 0:
        i -= 1
        n_f = n_freq_pattern[j]
        m_k = max_k_pattern[k]
        j += 1
        k += 1
        if j >= len(n_freq_pattern):
            j = 0
        if k >= len(max_k_pattern):
            k = 0
        try:
            y_filled, k_opt, freqs_opt, amps_opt = lomb_scargle_multi_interpolate_auto_k(
                t, y, t_interp,
                n_freq=n_f,
                max_k=m_k,
                criterion=criterion,
                f_min=f_min,
                f_max=f_max,
                normalization=normalization)
        except Exception as _:
            fail_early -= 1
            continue
        combined_t, combined_y = combine_original_and_interpolated(t, y, t_interp, y_filled)
        if max(combined_y) > max_y or min(combined_y) < min_y:
            #print(i, n_f, m_k, max(combined_y), max_y, min(combined_y), min_y)
            fail_early -= 1
            continue
        return y_filled, k_opt, freqs_opt, amps_opt, combined_t, combined_y, n_f, m_k
    return None, None, None, None, None, None, None, None


def fillMissingValues(df: pd.DataFrame) -> pd.DataFrame:
    from satoriengine.veda.interpolation.lomb_scargle import lomb_scargle_search
    t_interp = df.index.to_julian_date().values
    t_missing = df.dropna().index.to_julian_date().values
    y_missing = df["value"].dropna().values
    (
        y_filled, k_opt, freqs_opt, amps_opt, combined_t, combined_y, n_freq, max_k
    ) = lomb_scargle_search(t_missing, y_missing, t_interp, f_min=3, f_max=200)
    #visualizeToPng(t_interp, y_filled, annotations=None, output_path='./viz/original_inter.png')
    if combined_y is None:
        df["value"] = df["value"].fillna(method='ffill')
        return df
    df["value"] = combined_y
    return df

# Example usage
if __name__ == "__main__":
    import numpy as np

    # Example data with 1000 points combining two sine waves
    t = np.linspace(0, 999, 1000)
    rng = np.random.default_rng(seed=42)
    y = (
        np.sin(2.0 * np.pi * 0.005 * t) +  # First sine wave
        0.5 * np.sin(2.0 * np.pi * 0.01 * t) +  # Second sine wave
        0.3 * np.sin(2.0 * np.pi * 0.02 * t) +  # Third sine wave
        0.2 * np.sin(2.0 * np.pi * 0.03 * t) +  # Fourth sine wave
        0.1 * np.sin(2.0 * np.pi * 0.05 * t) +  # Fifth sine wave
        0.2 * rng.standard_normal(len(t))  # Noise
    )

    # Remove data points from index 600 to 800
    #t_missing = np.concatenate((t[:600], t[800:]))
    #y_missing = np.concatenate((y[:600], y[800:]))
    t_missing_multiple = np.concatenate((t[:200], t[270:500], t[550:700], t[750:]))
    y_missing_multiple = np.concatenate((y[:200], y[270:500], y[550:700], y[750:]))

    # Times to interpolate (fill gap from 600 to 800)
    t_interp = np.linspace(0, 999, 1000)

    # call directly for testing
    #y_filled, k_opt, freqs_opt, amps_opt = lomb_scargle_multi_interpolate_auto_k(
    #    t_missing, y_missing, t_interp,
    #    n_freq=2000,
    #    max_k=20,
    #    criterion='BIC')
    #
    # Visualize the original and interpolated data
    #visualizeToPng(t_missing, y_missing, annotations=None, output_path='./original_data.png')
    #visualizeToPng(t_interp, y_filled, annotations=None, output_path='./interpolated_data.png')
    #combined_t, combined_y = combine_original_and_interpolated(t_missing, y_missing, t_interp, y_filled)
    #visualizeToPng(combined_t, combined_y, annotations=None, output_path='./combined_data.png')

    # auto k and n_freq search
    #y_filled, k_opt, freqs_opt, amps_opt, combined_t, combined_y, n_freq, max_k = lomb_scargle_search(
    #    t_missing, y_missing, t_interp,
    #    n_freq=2000,
    #    max_k=20,
    #    criterion='BIC')
    #
    ## Visualize the original and interpolated data
    #visualizeToPng(t_missing, y_missing, annotations=None, output_path='./original_data.png')
    #visualizeToPng(t_interp, y_filled, annotations=None, output_path='./interpolated_data.png')
    #visualizeToPng(combined_t, combined_y, annotations=None, output_path='./combined_data.png')
    #
    #print(f"Optimal number of frequencies: {k_opt}")
    #print("Frequencies used:", freqs_opt)
    #print("Amplitude coefficients:", amps_opt)
    #print("Interpolated signal:", y_filled)

    # auto k and n_freq search
    y_filled, k_opt, freqs_opt, amps_opt, combined_t, combined_y, n_freq, max_k = lomb_scargle_search(
        t_missing_multiple, y_missing_multiple, t_interp)

    # Visualize the original and interpolated data
    visualizeToPng(t_missing_multiple, y_missing_multiple, annotations=None, output_path='./viz/original_multiple.png')
    visualizeToPng(t_interp, y_filled, annotations=None, output_path='./viz/interpolated_mutiple.png')
    visualizeToPng(combined_t, combined_y, annotations=None, output_path='./viz/combined_mutiple.png')

    print(f"Optimal number of frequencies: {k_opt}")
    print("Frequencies used:", freqs_opt)
    print("Amplitude coefficients:", amps_opt)
    print("Interpolated signal:", y_filled)
