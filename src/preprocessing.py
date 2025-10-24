import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2
from pathlib import Path
from scipy.signal import correlate
from scipy.signal import butter, filtfilt
from ahrs.filters import Madgwick
#from ahrs.common.orientation import q_rotate

# Define project root and data directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def get_participant_data_path(participant_type, participant_name):
    """
    Get the path to a participant's data directory.
    
    Parameters:
        participant_type (str): Either 'students' or 'older_adults'
        participant_name (str): Name of the participant
        
    Returns:
        Path: Path to the participant's data directory
    """
    return RAW_DATA_DIR / participant_type / participant_name


def get_processed_data_path(participant_type, participant_name):
    """
    Get the path to a participant's processed data directory.
    
    Parameters:
        participant_type (str): Either 'students' or 'older_adults'
        participant_name (str): Name of the participant
        
    Returns:
        Path: Path to the participant's processed data directory
    """
    return PROCESSED_DATA_DIR / participant_type / participant_name


def load_and_process_zed_data(zed_file_path):
    """
    Load and process ZED camera data.
    
    Parameters:
        zed_file_path (str or Path): Path to the ZED data file
        
    Returns:
        tuple: (zed_df, zed_com_df) - DataFrames with joint positions and COM positions
    """
    zed_df = pd.read_csv(zed_file_path)
    
    # Drop the first column of indexes
    zed_df.drop(columns=zed_df.columns[0], axis=1, inplace=True)
    
    # Replace headers
    zed_df.columns = ['time (s)',
                      'pelvis_X', 'pelvis_Y', 'pelvis_Z',
                      'naval_spine_X', 'naval_spine_Y', 'naval_spine_Z',
                      'left_hip_X', 'left_hip_Y', 'left_hip_Z',
                      'right_hip_X', 'right_hip_Y', 'right_hip_Z']
    
    # Convert from meters to centimeters
    coord_columns = [col for col in zed_df.columns if col != 'time (s)']
    zed_df[coord_columns] = zed_df[coord_columns] * 100

    zed_df['COM_ML'] = (zed_df['naval_spine_X'] + zed_df['left_hip_X'] + zed_df['right_hip_X']) / 3
    zed_df['COM_AP'] = (zed_df['naval_spine_Z'] + zed_df['left_hip_Z'] + zed_df['right_hip_Z']) / 3
    zed_df['COM_SI'] = (zed_df['naval_spine_Y'] + zed_df['left_hip_Y'] + zed_df['right_hip_Y']) / 3

    zed_com_df = zed_df[['time (s)', 'COM_ML', 'COM_AP', 'COM_SI']]
    return zed_df, zed_com_df


def load_and_process_force_plate_data(force_plate_file_dir, reverse_y_axis=False):
    force_plate_df = pd.read_csv(force_plate_file_dir, sep='\t', skiprows=8, encoding='latin1')
    force_plate_df.columns = force_plate_df.iloc[8]
    force_plate_df = force_plate_df.iloc[10:]
    force_plate_df.reset_index(drop=True, inplace=True)

    for col in force_plate_df.columns:
        force_plate_df[col] = pd.to_numeric(force_plate_df[col], errors='coerce')

    force_plate_df.rename(columns={'abs time (s)': 'time (s)'}, inplace=True)

    if reverse_y_axis:
        force_plate_df['Fy'] = -force_plate_df['Fy']
        force_plate_df['Ay'] = -force_plate_df['Ay']

    # Convert relevant columns to centimeters (e.g., position-based)
    position_cols = [col for col in force_plate_df.columns if col in ['Ax', 'Ay']]
    force_plate_df[position_cols] = force_plate_df[position_cols] * 100

    return force_plate_df


# --------------------------------------------------- phones ---------------------------------------------------

def load_and_process_phone_data(file_path):
    df = pd.read_csv(file_path)

    # Drop unnamed index column if it exists
    if df.columns[0].lower().startswith("unnamed"):
        df.drop(columns=df.columns[0], inplace=True)

    # Clean and standardize column names
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.rename(columns={
        'Time since start in ms': 'time_ms',
        'ACCELEROMETER X (m/s²)': 'acc_x',
        'ACCELEROMETER Y (m/s²)': 'acc_y',
        'ACCELEROMETER Z (m/s²)': 'acc_z',
        'GYROSCOPE X (rad/s)': 'gyro_x',
        'GYROSCOPE Y (rad/s)': 'gyro_y',
        'GYROSCOPE Z (rad/s)': 'gyro_z'
    }, inplace=True)

    # Convert time to seconds
    df['time'] = df['time_ms'] / 1000.0

    return df[['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]


# def apply_orientation_correction(df, fs=100):
#     acc = df[['acc_x', 'acc_y', 'acc_z']].values
#     gyr = df[['gyro_x', 'gyro_y', 'gyro_z']].values

#     # Initialize Madgwick filter
#     madgwick = Madgwick(sampleperiod=1.0/fs)

#     # Container for quaternions
#     quaternions = np.zeros((len(df), 4))
#     q = np.array([1.0, 0.0, 0.0, 0.0])  # initial quaternion

#     for t in range(len(df)):
#         q = madgwick.updateIMU(q, gyr[t], acc[t])
#         quaternions[t] = q

#     # Rotate accelerometer vectors to global frame
#     acc_global = np.array([q_rotate(q, a) for q, a in zip(quaternions, acc)])

#     # Add back to DataFrame
#     df[['acc_x_global', 'acc_y_global', 'acc_z_global']] = acc_global

#     return df


def estimate_delay_cross_correlation(sig1, sig2, fs=100, max_lag_seconds=3):
    n = min(len(sig1), len(sig2))
    sig1, sig2 = sig1[:n], sig2[:n]
    corr = correlate(sig1, sig2, mode='full')
    lags = np.arange(-n + 1, n)
    lag = lags[np.argmax(corr)]
    max_lag = int(fs * max_lag_seconds)
    lag = np.clip(lag, -max_lag, max_lag)
    return lag / fs


def butter_filter(data, fs, cutoff, btype='low', order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype=btype, analog=False)
    return filtfilt(b, a, data)


def sync_and_plot_phones(df_front, df_back, fs=100, threshold_factor=2.0, window=3.0, skip_seconds=0.0):
    # 1) Filter once (3–10 Hz) on the ORIGINAL rows
    acc_front_raw = df_front[['acc_x','acc_y','acc_z']].values
    acc_back_raw  = df_back [['acc_x','acc_y','acc_z']].values

    acc_front_filt = np.array([butter_filter(acc_front_raw[:, i], fs, 3,  btype='high') for i in range(3)]).T
    acc_back_filt  = np.array([butter_filter(acc_back_raw[:,  i], fs, 3,  btype='high') for i in range(3)]).T

    acc_front_filt = np.array([butter_filter(acc_front_filt[:, i], fs, 10, btype='low') for i in range(3)]).T
    acc_back_filt  = np.array([butter_filter(acc_back_filt[:,  i], fs, 10, btype='low') for i in range(3)]).T

    acc_mag_front0 = np.linalg.norm(acc_front_filt, axis=1)
    acc_mag_back0  = np.linalg.norm(acc_back_filt,  axis=1)
    t_front0 = df_front['time'].to_numpy()
    t_back0  = df_back ['time'].to_numpy()

    # 2) Estimate delay on the filtered magnitudes
    delay_sec = estimate_delay_cross_correlation(acc_mag_front0, acc_mag_back0, fs=fs)

    # 3) Shift times AND apply the SAME masks to the magnitude arrays (no re-filter)
    if delay_sec > 0:
        # back lags -> drop first delay seconds from BACK
        mask_b = t_back0 >= delay_sec
        t_b1   = t_back0[mask_b] - delay_sec
        b_mag1 = acc_mag_back0[mask_b]

        t_f1   = t_front0
        f_mag1 = acc_mag_front0
    elif delay_sec < 0:
        cut    = -delay_sec
        mask_f = t_front0 >= cut
        t_f1   = t_front0[mask_f] + delay_sec  # delay_sec negative
        f_mag1 = acc_mag_front0[mask_f]

        t_b1   = t_back0
        b_mag1 = acc_mag_back0
    else:
        t_f1, f_mag1 = t_front0, acc_mag_front0
        t_b1, b_mag1 = t_back0,  acc_mag_back0

    # 4) Keep only the overlap window in TIME, and trim mags with the SAME masks
    if len(t_f1) == 0 or len(t_b1) == 0:
        print(f"✅ Estimated delay between phones: {delay_sec:.2f} seconds")
        print("📍 Detected shared step time: None (no overlap)")
        return None, delay_sec

    tmin = max(t_f1[0], t_b1[0])
    tmax = min(t_f1[-1], t_b1[-1])
    mask_f2 = (t_f1 >= tmin) & (t_f1 <= tmax)
    mask_b2 = (t_b1 >= tmin) & (t_b1 <= tmax)

    t_f2, f_mag2 = t_f1[mask_f2], f_mag1[mask_f2]
    t_b2, b_mag2 = t_b1[mask_b2], b_mag1[mask_b2]

    # 5) Equalize lengths by truncation (same fs ⇒ near-identical grids)
    n = min(len(t_f2), len(t_b2), len(f_mag2), len(b_mag2))
    if n == 0:
        print(f"✅ Estimated delay between phones: {delay_sec:.2f} seconds")
        print("📍 Detected shared step time: None (no overlap after trim)")
        return None, delay_sec

    t    = t_f2[:n]          # choose front as the common axis
    fmag = f_mag2[:n]
    bmag = b_mag2[:n]
    avg  = 0.5 * (fmag + bmag)

    # 6) Step detection: skip by TIME, not samples
    start_idx = np.searchsorted(t, skip_seconds, side='left')
    seg = avg[start_idx:] if start_idx < len(avg) else avg[-1:]
    thr = seg.mean() + threshold_factor * seg.std()

    over = seg > thr
    if not np.any(over):
        spike_time = None
    else:
        spike_idx = start_idx + int(np.argmax(over))
        spike_time = float(t[spike_idx])

    # 7) Plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True, gridspec_kw={'width_ratios': [2, 1]})
    axs[0].plot(t, fmag, label='Front Phone')
    axs[0].plot(t, bmag, label='Back Phone')
    axs[0].axhline(thr, color='red', linestyle='--', label='Step Threshold')
    if spike_time is not None:
        axs[0].axvline(spike_time, color='green', linestyle='--', label=f'Step ≈ {spike_time:.2f}s')
    axs[0].set_title(f"Full Signal\nEstimated Delay: {delay_sec:.2f}s")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Acceleration Magnitude (m/s²)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, fmag, label='Front Phone')
    axs[1].plot(t, bmag, label='Back Phone')
    axs[1].axhline(thr, color='red', linestyle='--')
    if spike_time is not None:
        axs[1].axvline(spike_time, color='green', linestyle='--')
        axs[1].set_xlim(max(t[0], spike_time - window), min(t[-1], spike_time + window))
    axs[1].set_title(f'Zoomed View ±{window}s')
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(True)

    fig.suptitle("Phone Synchronization and Shared Step Detection", fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"✅ Estimated delay between phones: {delay_sec:.2f} seconds")
    print(f"📍 Detected shared step time: {spike_time if spike_time is None else f'{spike_time:.2f}'} seconds")

    return spike_time, delay_sec


def calculate_delay_and_step_time(df_front, df_back, fs=100, threshold_factor: float = 2.0, skip_seconds: float = 0.0):
    """
    Estimate delay between phones using 3–10 Hz filtered acceleration magnitudes.
    Detect a shared step time AFTER aligning the two phones by the delay.

    Args:
        df_front, df_back: DataFrames with columns ['time','acc_x','acc_y','acc_z'].
        fs: sampling rate (Hz)
        threshold_factor: step threshold = mean + threshold_factor * std (on averaged magnitude)
        skip_seconds: ignore the first 'skip_seconds' when searching for the step

    Returns:
        delay_sec (float), step_time_sec (float or None)
    """

    # Filter 3–10 Hz per axis, then magnitude (for delay estimation)
    acc_front_raw = df_front[['acc_x', 'acc_y', 'acc_z']].values
    acc_back_raw  = df_back [['acc_x', 'acc_y', 'acc_z']].values

    acc_front_filt = np.array([butter_filter(acc_front_raw[:, i], fs, 3,  btype='high') for i in range(3)]).T
    acc_back_filt  = np.array([butter_filter(acc_back_raw[:,  i], fs, 3,  btype='high') for i in range(3)]).T

    acc_front_filt = np.array([butter_filter(acc_front_filt[:, i], fs, 10, btype='low')  for i in range(3)]).T
    acc_back_filt  = np.array([butter_filter(acc_back_filt[:,  i], fs, 10, btype='low')  for i in range(3)]).T

    acc_mag_front = np.linalg.norm(acc_front_filt, axis=1)
    acc_mag_back  = np.linalg.norm(acc_back_filt,  axis=1)

    # Delay via cross-correlation on filtered magnitudes
    delay_sec = estimate_delay_cross_correlation(acc_mag_front, acc_mag_back, fs=fs)

    # Apply delay to the DataFrames' time axes and trim to overlap
    f, b = df_front.copy(), df_back.copy()
    if delay_sec > 0:
        # back lags -> drop its first 'delay' seconds and shift back in time
        b = b[b['time'] >= delay_sec].copy()
        b['time'] -= delay_sec
    elif delay_sec < 0:
        # front lags
        f = f[f['time'] >= -delay_sec].copy()
        f['time'] += delay_sec

    # overlap guard
    tmin = max(f['time'].min(), b['time'].min())
    tmax = min(f['time'].max(), b['time'].max())
    f = f[(f['time'] >= tmin) & (f['time'] <= tmax)].reset_index(drop=True)
    b = b[(b['time'] >= tmin) & (b['time'] <= tmax)].reset_index(drop=True)

    if len(f) == 0 or len(b) == 0:
        return delay_sec, None

    # Recompute 3–10 Hz magnitudes on aligned data (sample-true alignment)
    def filt_mag_3_10(df):
        raw = df[['acc_x','acc_y','acc_z']].values
        hp  = np.array([butter_filter(raw[:, i], fs, 3,  btype='high') for i in range(3)]).T
        lp  = np.array([butter_filter(hp[:,  i], fs, 10, btype='low')  for i in range(3)]).T
        return np.linalg.norm(lp, axis=1)

    f_mag = filt_mag_3_10(f)
    b_mag = filt_mag_3_10(b)

    n = min(len(f), len(b), len(f_mag), len(b_mag))
    if n == 0:
        return delay_sec, None

    t = f['time'].to_numpy()[:n]
    avgmag = 0.5 * (f_mag[:n] + b_mag[:n])

    # Detect shared step on averaged magnitude (mean + k*std after skipping)
    start_idx = int(skip_seconds * fs)
    if start_idx >= len(avgmag):
        return delay_sec, None

    seg = avgmag[start_idx:]
    thr = seg.mean() + threshold_factor * seg.std()
    idx_rel = np.argmax(seg > thr)
    if seg[idx_rel] <= thr:
        return delay_sec, None

    spike_idx  = start_idx + idx_rel
    step_time_sec = float(t[spike_idx])

    return delay_sec, step_time_sec


def calculate_delay_and_step_tables(
    group: str,
    participants: list,
    fs: int = 100,
    threshold_factor: float = 2.0,
    skip_seconds: float = 0.0):
    """
    Compute per-trial delay (sec) and shared step time (sec on aligned time base)
    between front/back phones.

    Returns:
        delay_table, step_table
        Each is a DataFrame indexed by participant with columns:
        open1..open5, closed1..closed5 (float seconds or NaN on failure)
    """
    trials = [f"{state}{i}" for state in ("open", "closed") for i in range(1, 6)]
    delay_table = pd.DataFrame(index=participants, columns=trials, dtype=float)
    step_table  = pd.DataFrame(index=participants, columns=trials, dtype=float)

    for participant in participants:
        for state in ("open", "closed"):
            for i in range(1, 6):
                trial = f"{state}{i}"

                front_path = os.path.join(
                    "..", "data", "raw", group,
                    participant, f"{participant}_front_phone",
                    f"{participant}_eyes_{state}{i}.csv"
                )
                back_path = os.path.join(
                    "..", "data", "raw", group,
                    participant, f"{participant}_back_phone",
                    f"{participant}_eyes_{state}{i}.csv"
                )

                try:
                    df_front = load_and_process_phone_data(front_path)
                    df_back  = load_and_process_phone_data(back_path)

                    delay, step_time = calculate_delay_and_step_time(
                        df_front, df_back,
                        fs=fs,
                        threshold_factor=threshold_factor,
                        skip_seconds=skip_seconds
                    )

                    delay_table.at[participant, trial] = round(float(delay), 2)
                    step_table.at[participant, trial]  = (
                        float("nan") if step_time is None else round(float(step_time), 2)
                    )

                except Exception:
                    delay_table.at[participant, trial] = float("nan")
                    step_table.at[participant, trial]  = float("nan")

    return delay_table, step_table


def extract_aligned_trimmed_filtered(
    group, participants, delay_table, step_table,
    fs=100, start_offset=5.0, duration=25.0,
    lowcut=0.5, highcut=10.0,
    save=True):
    """
    Align and trim original 3-axis accelerations using delay & step tables,
    filter each axis with butter_filter (band 0.5–10 Hz),
    and save to data/intermediate/ with the same structure as raw.

    Returns:
        dict[(participant, trial)] = {
            'front_df': aligned+trimmed+filtered DataFrame,
            'back_df' : aligned+trimmed+filtered DataFrame
        }
    """
    results = {}
    trials = [(state, i) for state in ("open", "closed") for i in range(1, 6)]

    for p in participants:
        for state, i in trials:
            trial = f"{state}{i}"
            if pd.isna(delay_table.at[p, trial]) or pd.isna(step_table.at[p, trial]):
                continue
            delay = float(delay_table.at[p, trial])
            step  = float(step_table.at[p, trial])

            try:
                # --- load original ---
                front_path = os.path.join("..","data","raw",group,
                                          p, f"{p}_front_phone", f"{p}_eyes_{state}{i}.csv")
                back_path  = os.path.join("..","data","raw",group,
                                          p, f"{p}_back_phone",  f"{p}_eyes_{state}{i}.csv")
                df_f = load_and_process_phone_data(front_path)
                df_b = load_and_process_phone_data(back_path)

                # --- filter each axis ---
                for axis in ["acc_x","acc_y","acc_z"]:
                    df_f[axis] = butter_filter(df_f[axis].values, fs, lowcut, btype="high")
                    df_f[axis] = butter_filter(df_f[axis].values, fs, highcut, btype="low")
                    df_b[axis] = butter_filter(df_b[axis].values, fs, lowcut, btype="high")
                    df_b[axis] = butter_filter(df_b[axis].values, fs, highcut, btype="low")

                # --- apply delay ---
                if delay > 0:
                    df_b = df_b[df_b['time'] >= delay].copy()
                    df_b['time'] -= delay
                elif delay < 0:
                    df_f = df_f[df_f['time'] >= -delay].copy()
                    df_f['time'] += delay

                # --- keep overlap only ---
                tmin = max(df_f['time'].iloc[0], df_b['time'].iloc[0])
                tmax = min(df_f['time'].iloc[-1], df_b['time'].iloc[-1])
                df_f = df_f[(df_f['time'] >= tmin) & (df_f['time'] <= tmax)].reset_index(drop=True)
                df_b = df_b[(df_b['time'] >= tmin) & (df_b['time'] <= tmax)].reset_index(drop=True)

                # --- trim window [step+5, step+30] ---
                start, end = step + start_offset, step + start_offset + duration
                mask_f = (df_f['time'] >= start) & (df_f['time'] < end)
                mask_b = (df_b['time'] >= start) & (df_b['time'] < end)
                df_f_win, df_b_win = df_f[mask_f].reset_index(drop=True), df_b[mask_b].reset_index(drop=True)

                # equalize length
                n = min(len(df_f_win), len(df_b_win))
                if n == 0:
                    continue
                df_f_win, df_b_win = df_f_win.iloc[:n], df_b_win.iloc[:n]

                results[(p, trial)] = {"front_df": df_f_win, "back_df": df_b_win}

                # --- save if requested ---
                if save:
                    # build intermediate paths
                    inter_base = os.path.join("..","..","data","intermediate",group,p)
                    front_dir = os.path.join(inter_base, f"{p}_front_phone")
                    back_dir  = os.path.join(inter_base, f"{p}_back_phone")
                    os.makedirs(front_dir, exist_ok=True)
                    os.makedirs(back_dir,  exist_ok=True)

                    front_out = os.path.join(front_dir, f"{p}_eyes_{state}{i}.csv")
                    back_out  = os.path.join(back_dir,  f"{p}_eyes_{state}{i}.csv")

                    df_f_win.to_csv(front_out, index=False)
                    df_b_win.to_csv(back_out,  index=False)

            except Exception:
                continue

# --------------------------------------------------------------------------------------------------------------

def plot_trim_point_decision(df, x_col, y_col, frames_num=50, find_recent=True, std_amount=2):
    # Calculate the mean and standard deviation of the first frames_num values
    initial_mean = df[y_col].iloc[:frames_num].mean()
    initial_std = df[y_col].iloc[:frames_num].std()
    upper_std = initial_mean + std_amount * initial_std
    lower_std = initial_mean - std_amount * initial_std
    
    # Find the crossing point based on the find_recent flag
    cross_index = None
    if find_recent:
        for i in reversed(range(1, len(df))):
            if (df[y_col].iloc[i] > upper_std and df[y_col].iloc[i - 1] <= upper_std) or (df[y_col].iloc[i] < lower_std and df[y_col].iloc[i - 1] >= lower_std):
                cross_index = i
                break
    else:
        for i in range(1, len(df)):
            if (df[y_col].iloc[i] > upper_std and df[y_col].iloc[i - 1] <= upper_std) or (df[y_col].iloc[i] < lower_std and df[y_col].iloc[i - 1] >= lower_std):
                cross_index = i
                break

    # Plot the data
    plt.figure(figsize=(16, 4))

    # Plot the initial part of the data
    if cross_index is not None:
        plt.plot(df[x_col].iloc[:cross_index], df[y_col].iloc[:cross_index], label='Data before trim')
        plt.plot(df[x_col].iloc[cross_index:], df[y_col].iloc[cross_index:], color='orange', label='Trimmed Data')
        plt.axvline(x=df[x_col].iloc[cross_index], color='g', linestyle='--', label='Trim Start')
    else:
        plt.plot(df[x_col], df[y_col], label='Data')

    # Plot the mean and standard deviation lines
    #plt.axhline(y=initial_mean, color='b', linestyle='-', label=f'Mean: {initial_mean:.2f}')
    plt.axhline(y=upper_std, color='r', linestyle='--', label=f'Mean + {std_amount}*STD')
    plt.axhline(y=lower_std, color='r', linestyle='--', label=f'Mean - {std_amount}*STD')

    plt.legend()
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{y_col} with {std_amount}*STD lines for the first {frames_num} samples')
    plt.show()
 

def show_decision_trim_plots(zed_file_dir, 
                             #force_plate_file_dir, 
                             zed_frames=50):
    zed_data, zed_com_data = load_and_process_zed_data(zed_file_dir)
    #force_plate_data = load_and_process_force_plate_data(force_plate_file_dir, reverse_y_axis=True)

    plot_trim_point_decision(zed_com_data, 'time (s)', 'COM_AP', frames_num=zed_frames)


def trim_dataframe_with_window(df, x_col, y_col, frames_num=50, start_offset=5, window_duration=30, find_recent=True, std_amount=2):
    """
    Trims the dataframe based on the crossing point of the upper standard deviation and 
    retains data within a specified time window.

    This function calculates the initial mean and standard deviation for the first 
    `frames_num` values of the specified `y_col`. It then finds the most recent 
    crossing point where the value goes from below to above the upper standard deviation (`upper_std`) or above to below. 
    The dataframe is trimmed to include only the data starting 5 seconds after the trimming point 
    and lasting for 30 seconds.

    Parameters:
    - df: pd.DataFrame - The input dataframe containing the data to be trimmed.
    - x_col: str - The name of the column representing the x-axis (typically time).
    - y_col: str - The name of the column representing the y-axis (the value to be analyzed).
    - frames_num: int - The number of initial frames used to calculate the mean and standard deviation (default is 50).
    - start_offset: int - The offset in seconds to start trimming after the trimming point (default is 5 seconds).
    - window_duration: int - The duration in seconds of the data window to retain after the start offset (default is 30 seconds).
    - find_recent: bool - If True, finds the most recent crossing point; if False, finds the first crossing point (default is True).
    
    Returns:
    - pd.DataFrame - A dataframe containing the data within the specified time window.
    """
    # Calculate the mean and standard deviation of the first frames_num values
    initial_mean = df[y_col].iloc[:frames_num].mean()
    initial_std = df[y_col].iloc[:frames_num].std()
    upper_std = initial_mean + std_amount * initial_std
    lower_std = initial_mean - std_amount * initial_std
    
        # Find the crossing point based on the find_recent flag
    cross_index = None
    if find_recent:
        for i in reversed(range(1, len(df))):
            if (df[y_col].iloc[i] > upper_std and df[y_col].iloc[i - 1] <= upper_std) or (df[y_col].iloc[i] < lower_std and df[y_col].iloc[i - 1] >= lower_std):
                cross_index = i
                break
    else:
        for i in range(1, len(df)):
            if (df[y_col].iloc[i] > upper_std and df[y_col].iloc[i - 1] <= upper_std) or (df[y_col].iloc[i] < lower_std and df[y_col].iloc[i - 1] >= lower_std):
                cross_index = i
                break

    # If no crossing point is found, return an empty DataFrame
    if cross_index is None:
        return pd.DataFrame(columns=df.columns)

    # Find the index corresponding to the start and end of the window
    start_time = df[x_col].iloc[cross_index] + start_offset
    end_time = start_time + window_duration

    # Select the data within the time window
    trimmed_data = df[(df[x_col] >= start_time) & (df[x_col] <= end_time)]

    return trimmed_data


def calculate_mad(df, column):
    """
    Calculate the Mean Absolute Deviation (MAD) of a specified column in a DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to calculate the MAD.

    Returns:
        float: The Mean Absolute Deviation (MAD) of the column.
    """
    mean_value = df[column].mean()
    deviations = df[column] - mean_value
    absolute_deviations = deviations.abs()
    mad = absolute_deviations.mean()
    return mad


def calculate_max_absolute_deviation(df, column):
    """
    Calculate the maximum absolute deviation from the mean of a specified column in a DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to calculate the maximum absolute deviation from the mean.

    Returns:
        float: The maximum absolute deviation from the mean of the column.
    """
    mean_value = df[column].mean()
    deviations = df[column] - mean_value
    absolute_deviations = deviations.abs()
    max_absolute_deviation = absolute_deviations.max()
    return max_absolute_deviation


def calculate_sda_ellipse_area(df, x_col, y_col):
    """
    Calculate the ellipse area that contains 95% of the data points in the given x and y columns.
    
    Parameters:
        df (DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column to use for the x-axis.
        y_col (str): The name of the column to use for the y-axis.
    
    Returns:
        float: The area of the 95% data ellipse.
    """
    # Calculate the covariance matrix
    cov_matrix = np.cov(df[x_col], df[y_col])
    
    # Calculate the eigenvalues of the covariance matrix (semi-major and semi-minor axis lengths)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    
    # Chi-squared quantile for 95% containment in 2D data (degrees of freedom = 2)
    chi2_quantile = chi2.ppf(0.95, 2)
    
    # The area of the ellipse is pi * sqrt(eigenvalue_1) * sqrt(eigenvalue_2) * chi-squared quantile
    ellipse_area = np.pi * chi2_quantile * np.sqrt(eigenvalues[0]) * np.sqrt(eigenvalues[1])
    
    return ellipse_area


def calculate_path_length(df, x_col, y_col):
    """
    Calculate the path length as the sum of Euclidean distances 
    between consecutive points in the specified x and y columns.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column representing the x-coordinates.
        y_col (str): The name of the column representing the y-coordinates.

    Returns:
        float: The total path length.
    """
    # Calculate the differences between consecutive points for x and y
    x_diff = df[x_col].diff()
    y_diff = df[y_col].diff()

    # Calculate Euclidean distances between consecutive points
    distances = np.sqrt(x_diff**2 + y_diff**2)

    # Sum the distances to get the path length
    path_length = distances.sum()

    return path_length


def calculate_axis_rms(series):
    """Compute RMS for a single axis (e.g., AP or ML)."""
    mean = np.mean(series)
    return np.sqrt(np.mean((series - mean)**2))


def calculate_sway_rms(df, ap_col, ml_col):
    """
    Calculate AP RMS, ML RMS, and total sway RMS for a trial.
    
    Parameters:
    - df: DataFrame containing AP and ML CoP data
    - ap_col: column name for AP direction
    - ml_col: column name for ML direction
    
    Returns:
    - Dictionary with 'ap_rms', 'ml_rms', and 'sway_rms'
    """
    ap_rms = calculate_axis_rms(df[ap_col])
    ml_rms = calculate_axis_rms(df[ml_col])
    
    # Total sway RMS from resultant displacement
    rd = np.sqrt((df[ap_col] - df[ap_col].mean())**2 + (df[ml_col] - df[ml_col].mean())**2)
    sway_rms = np.sqrt(np.mean(rd**2))
    
    return ap_rms, ml_rms, sway_rms


def calculate_measurements(df, ml_col, ap_col):
    # Max distances
    ml_range = max(df[ml_col]) - min(df[ml_col])
    ap_range = max(df[ap_col]) - min(df[ap_col])
    
    # Range ratio
    range_ratio = ml_range/ap_range
 
    # MAD - Mean absolute deviation
    ml_mad = calculate_mad(df, ml_col)
    ap_mad = calculate_mad(df, ap_col)
    
    # Maximal deviation
    max_abs_dev_ml = calculate_max_absolute_deviation(df, ml_col)
    max_abs_dev_ap = calculate_max_absolute_deviation(df, ap_col)
    
    # Elliptical area
    ellipse_area = calculate_sda_ellipse_area(df, ml_col, ap_col)
    
    # Path length
    path_length = calculate_path_length(df, ml_col, ap_col)
    
    # Sway RMS
    ap_rms, ml_rms, sway_rms = calculate_sway_rms(df=df, ml_col=ml_col, ap_col=ap_col)

    return ml_range, ap_range, range_ratio,\
            ml_mad, ap_mad,\
            max_abs_dev_ml, max_abs_dev_ap, ellipse_area,\
            path_length, ml_rms, ap_rms, sway_rms


def calculate_phone_measurements(df, ml_col, ap_col):
    # Max distances
    ml_range = max(df[ml_col]) - min(df[ml_col])
    ap_range = max(df[ap_col]) - min(df[ap_col])
    
    # MAD - Mean absolute deviation
    ml_mad = calculate_mad(df, ml_col)
    ap_mad = calculate_mad(df, ap_col)
    
    # Maximal deviation
    max_abs_dev_ml = calculate_max_absolute_deviation(df, ml_col)
    max_abs_dev_ap = calculate_max_absolute_deviation(df, ap_col)
    
    # Sway RMS
    ap_rms, ml_rms, sway_rms = calculate_sway_rms(df=df, ml_col=ml_col, ap_col=ap_col)

    return ml_range, ap_range,\
            ml_mad, ap_mad,\
            max_abs_dev_ml, max_abs_dev_ap,\
            ml_rms, ap_rms, sway_rms


def build_phones_dataset(participant_type: str = "older_adults",
                               base_dir: str = os.path.join("..","..","data","intermediate")) -> pd.DataFrame:
    """
    Iterate over data/intermediate/<participant_type>/ and compute phone metrics per file.
    Output columns: ['participant name','state','trial','device','metric','value']

    - state: 'open' or 'closed'
    - trial: 1..5
    - device: 'front' or 'back'
    Metrics (exact labels):
      "AP Range", "ML Range", "AP MAD", "ML MAD",
      "AP RMS", "ML RMS", "AP Max abs dev", "ML Max abs dev", "Sway RMS"
    """
    root = os.path.join(base_dir, participant_type)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    records = []

    for participant_name in os.listdir(root):
        participant_path = os.path.join(root, participant_name)
        if not os.path.isdir(participant_path):
            continue

        # device folders: "<participant>_front_phone" / "<participant>_back_phone"
        for device_folder in os.listdir(participant_path):
            dev_path = os.path.join(participant_path, device_folder)
            if not os.path.isdir(dev_path):
                continue

            if device_folder.endswith("_front_phone"):
                device = "front"
            elif device_folder.endswith("_back_phone"):
                device = "back"
            else:
                continue  # skip unexpected folders

            for fname in os.listdir(dev_path):
                if not fname.endswith(".csv"):
                    continue

                trial_base = os.path.splitext(fname)[0]  # e.g., "ann_eyes_open1"
                # parse state and trial number
                state, trial = None, None
                if "eyes_open" in trial_base:
                    state = "open"
                    try:
                        trial = int(trial_base.split("eyes_open")[-1])
                    except Exception:
                        pass
                elif "eyes_closed" in trial_base:
                    state = "closed"
                    try:
                        trial = int(trial_base.split("eyes_closed")[-1])
                    except Exception:
                        pass
                if state is None or trial is None:
                    continue

                fpath = os.path.join(dev_path, fname)
                try:
                    df = pd.read_csv(fpath)
                    # require axis columns
                    if not {"acc_x", "acc_z"}.issubset(df.columns):
                        continue

                    (ml_range, ap_range,
                     ml_mad, ap_mad,
                     max_abs_dev_ml, max_abs_dev_ap,
                     ml_rms, ap_rms, sway_rms) = calculate_phone_measurements(
                        df, ml_col="acc_x", ap_col="acc_z"
                    )

                    metric_map = [
                        ("AP Range",        ap_range),
                        ("ML Range",        ml_range),
                        ("AP MAD",          ap_mad),
                        ("ML MAD",          ml_mad),
                        ("AP RMS",          ap_rms),
                        ("ML RMS",          ml_rms),
                        ("AP Max abs dev",  max_abs_dev_ap),
                        ("ML Max abs dev",  max_abs_dev_ml),
                        ("Sway RMS",        sway_rms),
                    ]

                    for metric_name, val in metric_map:
                        records.append({
                            "participant name": participant_name,
                            "state": state,
                            "trial": trial,
                            "device": device,
                            "metric": metric_name,
                            "value": float(val),
                        })

                except Exception:
                    # skip problematic files but continue
                    continue

    out = pd.DataFrame.from_records(records,
                                    columns=["participant name","state","trial","device","metric","value"])
    if not out.empty:
        out.sort_values(by=["participant name","state","trial","device","metric"], inplace=True, ignore_index=True)
    return out


def load_and_process(zed_file_dir, force_plate_file_dir, zed_frames=50):
    
    # Step 1: preprocess the raw data
    zed_df, zed_com_df = load_and_process_zed_data(zed_file_dir)
    force_plate_df = load_and_process_force_plate_data(force_plate_file_dir, reverse_y_axis=True)
    
    # Step 2: trim dataframes base on the 2 std logic
    zed_com_df_trimmed = trim_dataframe_with_window(zed_com_df, 'time (s)', 'COM_AP', frames_num=zed_frames)
    force_plate_df_trimmed = trim_dataframe_with_window(force_plate_df, 'time (s)', 'Fz', frames_num=50)

    # Step 3: calculate measurments
    zed_com_measures = calculate_measurements(df=zed_com_df_trimmed, ml_col='COM_ML', ap_col='COM_AP') 
    force_plate_measures = calculate_measurements(df=force_plate_df_trimmed, ml_col='Ax', ap_col='Ay')

    data = {
        'Measurement': ['ML Range', 'AP Range', 'Range Ratio',
                        'ML MAD', 'AP MAD', 'ML Max abs dev', 'AP Max abs dev', 
                        'Ellipse area', 'Path length', 'ML RMS', 'AP RMS', 'Sway RMS'],
        'ZED_COM': zed_com_measures,
        'Force_Plate': force_plate_measures
    }
    
    measurements_df = pd.DataFrame(data)

    return measurements_df


def process_all_experiments(participant_type, zed_frames_dict, window_duration=30):
    """
    Process all participant data for a specific participant type and combine results into a single DataFrame.

    Parameters:
        participant_type (str): Either 'students' or 'older_adults'
        zed_frames_dict (dict): A dictionary mapping participant and state combinations to specific ZED frame values.

    Returns:
        DataFrame: A combined DataFrame with all participant measurements.
    """
    print("="*50)
    print(f"Starting process_all_experiments with participant_type: {participant_type}")
    print(f"zed_frames_dict keys: {list(zed_frames_dict.keys())}")
    print("="*50)
    
    all_data = []
    participant_dir = RAW_DATA_DIR / participant_type
    
    print(f"Processing data from: {participant_dir}")
    
    if not participant_dir.exists():
        raise FileNotFoundError(f"Directory not found: {participant_dir}")

    # Iterate through all participant folders
    for participant_name in os.listdir(participant_dir):
        participant_path = participant_dir / participant_name
        
        if not participant_path.is_dir():
            continue
            
        print(f"\nProcessing participant: {participant_name}")
        
        # Look for "_zed" and "_force_plate" folders
        zed_dirs = [d for d in os.listdir(participant_path) if "_zed" in d]
        force_plate_dirs = [d for d in os.listdir(participant_path) if "_force_plate" in d]
        
        if not zed_dirs:
            print(f"  No ZED data found for {participant_name}")
            continue
            
        if not force_plate_dirs:
            print(f"  No force plate data found for {participant_name}")
            continue

        # Process each pair of `_zed` and `_force_plate` folders
        for zed_dir in zed_dirs:
            zed_folder_path = participant_path / zed_dir
            print(f"  Processing ZED folder: {zed_dir}")

            # Find the corresponding `_force_plate` folder
            force_plate_dir = zed_dir.replace("_zed", "_force_plate")
            force_plate_folder_path = participant_path / force_plate_dir

            if not force_plate_folder_path.exists():
                print(f"  Force plate folder not found: {force_plate_dir}")
                continue

            # Get list of files in both directories
            zed_files = sorted([f for f in os.listdir(zed_folder_path) if f.endswith('.csv')])
            force_plate_files = sorted([f for f in os.listdir(force_plate_folder_path) if f.endswith('.txt')])
            
            if not zed_files or not force_plate_files:
                print(f"  No matching files found in directories:")
                print(f"    ZED files: {len(zed_files)}")
                print(f"    Force plate files: {len(force_plate_files)}")
                continue

            # Process files inside each folder
            for zed_file, force_plate_file in zip(zed_files, force_plate_files):
                print(f"    Processing files: {zed_file} and {force_plate_file}")
                
                # Extract the state from the filename (e.g., "ann_eyes_open1.csv" -> "ann_eyes_open1")
                filename_without_ext = os.path.splitext(zed_file)[0]
                
                # Get the zed_frames value from the dictionary, defaulting to 50 if not found
                zed_frames = zed_frames_dict.get(filename_without_ext, 50)
                print(f"    Using {zed_frames} frames for {filename_without_ext}")

                # Process the data
                zed_file_path = zed_folder_path / zed_file
                force_plate_file_path = force_plate_folder_path / force_plate_file

                try:
                    # Process ZED data
                    zed_df, zed_com_df = load_and_process_zed_data(str(zed_file_path))
                    
                    # Process force plate data
                    force_plate_df = load_and_process_force_plate_data(str(force_plate_file_path), reverse_y_axis=True)
                    
                    # Trim the dataframes
                    zed_com_df_trimmed = trim_dataframe_with_window(df=zed_com_df, x_col='time (s)', 
                                                                    y_col='COM_AP', window_duration=window_duration, 
                                                                    frames_num=zed_frames)
                    force_plate_df_trimmed = trim_dataframe_with_window(df=force_plate_df, x_col='time (s)', 
                                                                        y_col='Fz', window_duration=window_duration, 
                                                                        frames_num=50)
                    
                    # Calculate measurements
                    zed_com_measures = calculate_measurements(df=zed_com_df_trimmed, ml_col='COM_ML', ap_col='COM_AP')
                    force_plate_measures = calculate_measurements(df=force_plate_df_trimmed, ml_col='Ax', ap_col='Ay')

                    parts = filename_without_ext.split('_')
                    state_trial = parts[-1]  # e.g., "open1" or "closed2"

                    if state_trial.startswith("open"):
                        state_name = "open"
                        trial_number = state_trial.replace("open", "")
                    elif state_trial.startswith("closed"):
                        state_name = "closed"
                        trial_number = state_trial.replace("closed", "")
                    else:
                        raise ValueError(f"Unexpected trial format in file: {zed_file}")


                    # Create measurements DataFrame
                    measurements_data = {
                        'Measurement': ['ML Range', 'AP Range', 'Range Ratio',
                                        'ML MAD', 'AP MAD', 'ML Max abs dev', 'AP Max abs dev', 
                                        'Ellipse area', 'Path length', 'ML RMS', 'AP RMS', 'Sway RMS'],
                        'ZED_COM': zed_com_measures,
                        'Force_Plate': force_plate_measures
                    }
                    measurements_df = pd.DataFrame(measurements_data)

                    # Reshape the data to the desired format
                    for _, row in measurements_df.iterrows():
                        for device, value in zip(["ZED_COM", "Force_Plate"], [row["ZED_COM"], row["Force_Plate"]]):
                            all_data.append({
                                "participant name": participant_name,
                                "state": state_name,
                                "trial": trial_number,
                                "device": device,
                                "metric": row["Measurement"],
                                "value": value
                            })
                except Exception as e:
                    print(f"    Error processing files: {str(e)}")
                    continue

    # Combine all collected data into a DataFrame
    if not all_data:
        print("No data was processed successfully!")
        return pd.DataFrame()
        
    combined_df = pd.DataFrame(all_data)
    print(f"\nSuccessfully processed {len(combined_df)} measurements")
    
    # Save the processed data directly in the participant type folder
    processed_dir = PROCESSED_DATA_DIR / participant_type
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_file = processed_dir / "measurements.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"Saved results to: {output_file}")

    return combined_df


def get_total_time_per_trial(participant_type, zed_frames_dict):
    """
    For each participant and trial, computes the total time (s) recorded
    in the trimmed ZED and force plate data.

    Returns:
        pd.DataFrame with columns: ['participant name', 'trial name', 'total_time_zed', 'total_time_fp']
    """
    print("="*50)
    print(f"Extracting total time values for participant_type: {participant_type}")
    print("="*50)

    time_data = []
    participant_dir = RAW_DATA_DIR / participant_type

    if not participant_dir.exists():
        raise FileNotFoundError(f"Directory not found: {participant_dir}")

    for participant_name in os.listdir(participant_dir):
        participant_path = participant_dir / participant_name

        if not participant_path.is_dir():
            continue

        zed_dirs = [d for d in os.listdir(participant_path) if "_zed" in d]
        force_plate_dirs = [d for d in os.listdir(participant_path) if "_force_plate" in d]

        if not zed_dirs or not force_plate_dirs:
            continue

        for zed_dir in zed_dirs:
            zed_folder_path = participant_path / zed_dir
            force_plate_dir = zed_dir.replace("_zed", "_force_plate")
            force_plate_folder_path = participant_path / force_plate_dir

            if not force_plate_folder_path.exists():
                continue

            zed_files = sorted([f for f in os.listdir(zed_folder_path) if f.endswith('.csv')])
            force_plate_files = sorted([f for f in os.listdir(force_plate_folder_path) if f.endswith('.txt')])

            for zed_file, force_plate_file in zip(zed_files, force_plate_files):
                trial_name = os.path.splitext(zed_file)[0]
                zed_frames = zed_frames_dict.get(trial_name, 50)

                try:
                    zed_file_path = zed_folder_path / zed_file
                    force_plate_file_path = force_plate_folder_path / force_plate_file

                    zed_df, zed_com_df = load_and_process_zed_data(str(zed_file_path))
                    force_plate_df = load_and_process_force_plate_data(str(force_plate_file_path), reverse_y_axis=True)

                    zed_com_df_trimmed = trim_dataframe_with_window(zed_com_df, 'time (s)', 'COM_AP', frames_num=zed_frames)
                    force_plate_df_trimmed = trim_dataframe_with_window(force_plate_df, 'time (s)', 'Fz', frames_num=50)

                    total_time_zed = zed_com_df_trimmed["time (s)"].max()-zed_com_df_trimmed["time (s)"].min()
                    total_time_fp = force_plate_df_trimmed["time (s)"].max()-force_plate_df_trimmed["time (s)"].min()

                    time_data.append({
                        "participant name": participant_name,
                        "trial name": trial_name,
                        "total_time_zed": total_time_zed,
                        "total_time_fp": total_time_fp
                    })

                except Exception as e:
                    print(f"Error processing {zed_file} / {force_plate_file}: {str(e)}")
                    continue

    return pd.DataFrame(time_data)


def get_total_time_per_trial_phones(participant_type: str) -> pd.DataFrame:
    """
    For each participant and trial in the intermediate folder, compute the total time (s).
    It uses the 'time' column in each file (max - min).

    Returns:
        pd.DataFrame with columns: 
            ['participant name', 'phone', 'trial name', 'total_time']
    """
    print("="*50)
    print(f"Extracting total time values from intermediate for participant_type: {participant_type}")
    print("="*50)

    time_data = []
    INTERMEDIATE_DIR = os.path.join("..", "..", "data", "intermediate")
    participant_dir = os.path.join(INTERMEDIATE_DIR, participant_type)

    if not os.path.exists(participant_dir):
        raise FileNotFoundError(f"Directory not found: {participant_dir}")

    for participant_name in os.listdir(participant_dir):
        participant_path = os.path.join(participant_dir, participant_name)
        if not os.path.isdir(participant_path):
            continue

        for phone_dir in os.listdir(participant_path):
            phone_path = os.path.join(participant_path, phone_dir)
            if not os.path.isdir(phone_path):
                continue

            for file_name in os.listdir(phone_path):
                if not file_name.endswith(".csv"):
                    continue

                trial_name = os.path.splitext(file_name)[0]
                file_path = os.path.join(phone_path, file_name)

                try:
                    df = pd.read_csv(file_path)
                    if "time" not in df.columns:
                        raise KeyError(f"'time' column missing in {file_name}")

                    total_time = df["time"].iloc[-1] - df["time"].iloc[0]

                    time_data.append({
                        "participant name": participant_name,
                        "phone": phone_dir,
                        "trial name": trial_name,
                        "total_time": total_time
                    })

                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    continue

    return pd.DataFrame(time_data)


def process_and_plot_fps(base_root1, base_root2):
    def process_fps(base_root):
        results = []
        for root, dirs, files in os.walk(base_root):
            # Looking for folders that end with _zed
            if os.path.basename(root).endswith("_zed"):
                participant_id = root.split(os.sep)[-2]  # get parent folder (participant)
                for file in files:
                    if file.endswith(".csv"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                num_rows = sum(1 for _ in f) - 1  # exclude header
                            fps = num_rows / 40  # 40 seconds trial
                            results.append({
                                "Participant": participant_id,
                                "File": file,
                                "Frames": num_rows,
                                "FPS": round(fps, 2),
                                "Group": os.path.basename(base_root)
                            })
                        except Exception as e:
                            print(f"❌ Error processing {file_path}: {e}")
        return pd.DataFrame(results)
    
    df1 = process_fps(base_root1)
    df2 = process_fps(base_root2)
    
    df = pd.concat([df1, df2], ignore_index=True)
    
    if df.empty:
        print("No valid CSV files found.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="FPS", y="Group", data=df, inner=None, palette="pastel", linewidth=0.6)
    sns.boxplot(x="FPS", y="Group", data=df, width=0.2, color="gray", fliersize=0)
    sns.stripplot(x="FPS", y="Group", data=df, size=5, color="black", alpha=0.5, jitter=0.25)

    plt.title("Raincloud Plot of ZED FPS Values by Group", fontsize=14)
    plt.xlabel("Frames Per Second (FPS)")
    plt.ylabel("Participant Group")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()