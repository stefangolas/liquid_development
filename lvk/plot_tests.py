import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_p1_curves_with_color_coding(file_path):
    """
    Loads test history data from a JSON file, plots the P1 curve for each test,
    and its Fourier transform, coloring the curve based on accuracy.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return

    # --- Configuration ---
    ACCURACY_THRESHOLD = 80.0
    ACCURATE_COLOR = 'red'
    INACCURATE_COLOR = 'grey'
    # ---------------------

    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    if not isinstance(data, list):
        print("Error: The JSON file content is not a list of test records.")
        return

    print("--- Test Accuracy Results ---")

    for i, entry in enumerate(data):
        test_number = i + 1

        # Accuracy → color
        plot_color = 'blue' # default
        accuracy_status = 'N/A'
        accuracy = entry.get('accuracy_percent')
        volume = entry.get('volume_uL', 'N/A')

        try:
            if isinstance(accuracy, (int, float)):
                if accuracy >= ACCURACY_THRESHOLD:
                    accuracy_status = 'ACCURATE'
                    plot_color = ACCURATE_COLOR
                else:
                    accuracy_status = 'INACCURATE'
                    plot_color = INACCURATE_COLOR

            print(f"Test {test_number} (Volume: {volume} uL): Accuracy = {accuracy}% ({accuracy_status})")

        except Exception as e:
            print(f"Warning: Could not process accuracy for Test {test_number}. Error: {e}")

        # Plot P1 + FFT
        try:
            p1_data = np.array(entry['tadm_data']['P1'])
            n = len(p1_data)

            # --- Time domain ---
            ax_time.plot(p1_data, label=f"Test {test_number}", color=plot_color, alpha=0.8, linewidth=1.5)

            # --- Frequency domain (FFT) ---
            fft_vals = np.fft.fft(p1_data)
            freqs = np.fft.fftfreq(n, d=1)  # assumes sampling interval = 1
            # Use only the positive frequencies
            pos_mask = freqs >= 0
            ax_freq.plot(freqs[pos_mask], np.abs(fft_vals[pos_mask]), 
                         label=f"Test {test_number}", color=plot_color, alpha=0.8, linewidth=1.5)

        except KeyError:
            print(f"Warning: Missing 'P1' data in test {test_number}. Skipping.")
        except Exception as e:
            print(f"Unexpected error in Test {test_number}: {e}")

    print("-----------------------------\n")

    # --- Time plot settings ---
    ax_time.set_title("P1 Curves (Time Domain)")
    ax_time.set_xlabel("Data Point Index (Time Step)")
    ax_time.set_ylabel("P1 Sensor Value")

    # --- Frequency plot settings ---
    ax_freq.set_title("Fourier Transforms of P1 Curves (Frequency Domain)")
    ax_freq.set_xlabel("Frequency (Hz, assuming Δt=1)")
    ax_freq.set_ylabel("Magnitude")

    # Custom legend
    proxy_accurate = Line2D([0], [0], color=ACCURATE_COLOR, lw=3)
    proxy_inaccurate = Line2D([0], [0], color=INACCURATE_COLOR, lw=3)

    custom_handles = [proxy_accurate, proxy_inaccurate]
    custom_labels = [f'Accurate (>= {ACCURACY_THRESHOLD}%)', f'Inaccurate (< {ACCURACY_THRESHOLD}%)']

    ax_time.legend(handles=custom_handles, labels=custom_labels,
                   bbox_to_anchor=(1.05, 1), loc='upper left', title='Accuracy')
    
    ax_time.grid(True, linestyle='--', alpha=0.6)
    ax_freq.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


# --- Run the script ---
plot_p1_curves_with_color_coding('test_history.json')
