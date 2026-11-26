"""Convert the binary iris codes in data/Iris_binary.xlsx into a .mat file.

Each row of the spreadsheet is one iris code: 10,240 space-separated bits.
The output holds a single uint8 array of shape (rows, 10240), which the
compression tool in src/ can read.

Usage:
    python scripts/xlsx_to_mat.py data/Iris_binary.xlsx data/iris_codes.mat
"""
import sys

import numpy as np
import openpyxl
import scipy.io


def main(xlsx_path, mat_path):
    sheet = openpyxl.load_workbook(xlsx_path, read_only=True).active
    rows = [str(r[0]).split() for r in sheet.iter_rows(values_only=True) if r[0] is not None]
    codes = np.array(rows, dtype=np.uint8)
    scipy.io.savemat(mat_path, {"iris_codes": codes})
    print(f"Saved {codes.shape[0]} iris codes of {codes.shape[1]} bits to {mat_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python scripts/xlsx_to_mat.py <input.xlsx> <output.mat>")
    main(sys.argv[1], sys.argv[2])
