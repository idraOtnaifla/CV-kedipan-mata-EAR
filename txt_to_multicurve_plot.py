import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def load_xy_from_txt(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    if data.ndim == 1:
        if data.size != 2:
            raise ValueError("File must contain two columns of X and Y values")
        return np.array([data[0]]), np.array([data[1]])
    if data.shape[1] != 2:
        raise ValueError("File must contain exactly two columns: X and Y")
    return data[:, 0], data[:, 1]


def plot_files(folder_path, file_pattern, output_filename):
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    files = sorted(glob.glob(os.path.join(folder_path, file_pattern)))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern '{file_pattern}' in {folder_path}")

    if len(files) != 10:
        print(f"Warning: expected 10 files, found {len(files)}. Plotting all available files.")

    plt.figure(figsize=(10, 6))
    for file_path in files:
        try:
            x, y = load_xy_from_txt(file_path)
            label = os.path.basename(file_path)
            plt.plot(x, y, label=label)
            print(f"Plotted: {label}")
        except Exception as exc:
            print(f"Skipping {file_path}: {exc}")

    plt.title("Multicurve Plot of Tab-Separated Data")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.grid(True)
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    output_path = os.path.join(folder_path, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load tab-separated X and Y values from text files and save a multicurve plot in the same folder."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="DATA/VIDEOS/OUTPUTS/hasil 10 blink 3 count/10 blink 3 count50 cm",
        help="Folder containing the .txt files",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for text files (default: *.txt)",
    )
    parser.add_argument(
        "--output",
        default="multicurve_plot.png",
        help="Output image filename (saved in the same folder)",
    )
    args = parser.parse_args()

    plot_files(args.folder, args.pattern, args.output)
