# Iris Biometric Compression

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-SciPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ROC%20%2F%20EER-F7931E?logo=scikitlearn&logoColor=white)

Lossless compression of binary iris biometric templates using **Huffman coding with LRP frequency normalisation**, plus a genuine-vs-impostor **Equal Error Rate (EER)** evaluation.

*B.Tech Computer Science & Engineering capstone project · SRM University-AP · 2025*

---

## Contents

- [Overview](#overview)
- [How it works](#how-it-works)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Team](#team)

---

## Overview

Biometric systems store a template for every enrolled person. Iris templates are large bit strings, so storing and transmitting them, especially for encrypted (privacy-preserving) matching, is costly. This project builds a compression tool for iris templates that:

- compresses `.mat` biometric files **without losing any information**, so the decompressed template is identical to the original
- adds an **enhanced LRP** step that reshapes the symbol frequencies before the Huffman tree is built
- compares compressed templates and evaluates matching performance with **ROC curves and Equal Error Rate**

The project's wider aim was to make templates small enough for homomorphic-encryption-based matching. This repository contains the compression and evaluation stage.

## How it works

```
Iris codes (.mat) ──► Byte frequencies ──► Enhanced LRP (optional) ──► Huffman tree
                                                                         │
     Compressed .npz  ◄── Pack bits + Huffman codes + shape/dtype ◄──────┘
            │
            ├──► Decompress ──► identical .mat file
            └──► Compare / EER evaluation ──► ROC curve with EER point
```

| Step | What happens |
| --- | --- |
| **1. Frequency analysis** | Count how often each byte value occurs in the template |
| **2. Enhanced LRP** *(optional)* | Combine a random projection of the frequencies (dimension 32) with local response normalisation over an adaptive window, then rescale to the original total |
| **3. Huffman coding** | Build a min-heap of symbols, merge into a Huffman tree, and assign variable-length prefix codes |
| **4. Packing** | Pad the bit string to whole bytes and save it with the code table, array shape, data type and variable name in a compressed `.npz` |
| **5. Decompression** | Rebuild the bit string, decode it with the reverse code table, and restore the original `.mat` file |
| **6. Evaluation** | Score genuine (original vs. compressed) and impostor (file vs. file) pairs, then plot the ROC curve and mark the EER and threshold |

The tool runs as an interactive menu: analyse, compress, decompress, compare two compressed files, and evaluate EER.

## Results

Measured on the 540 iris codes in [`data/Iris_binary.xlsx`](data/Iris_binary.xlsx) (10,240 bits each):

| Measure | Value |
| --- | --- |
| Original `.mat` file | 5,529,800 bytes |
| Compressed `.npz` file | 14,645 bytes |
| Compression ratio | **377.6 : 1** |
| Decompressed vs. original | Identical (lossless) |

Each value in an iris code is a 0 or 1, so Huffman coding stores it in 1 bit instead of 8, an 8:1 reduction. The `.npz` container then compresses the long runs of identical bits much further.

## Repository structure

```
iris-biometric-compression/
├── src/
│   ├── iris_compression.py    Final version: Huffman + enhanced LRP + EER evaluation
│   └── huffman_baseline.py    First version: plain Huffman compression of .mat files
├── scripts/
│   └── xlsx_to_mat.py         Converts the iris spreadsheet to a .mat file
├── data/
│   └── Iris_binary.xlsx       540 binary iris codes, 10,240 bits each
└── requirements.txt
```

## Getting started

```bash
# Clone and install
git clone https://github.com/abhiramjammula-AJ/iris-biometric-compression.git
cd iris-biometric-compression
pip install -r requirements.txt

# Convert the iris spreadsheet to a .mat file
python scripts/xlsx_to_mat.py data/Iris_binary.xlsx data/iris_codes.mat

# Run the tool and follow the menu
python src/iris_compression.py
```

Example session: choose **2** and enter `data/iris_codes.mat` to compress it, then **3** with `data/iris_codes_compressed.npz` to decompress it. For **5** (EER evaluation), give two or more `.mat` files.

> **Fixes since submission:** the decompressor now converts the stored data type to a string, which current NumPy versions require, and the EER summary uses a valid number format. The compression logic is unchanged.

## Team

**Abhiram Jammula**, Revanth Idupulapati, Harshini Bapanpalli and Hazeesh Shaik

Supervised by **Prof. Dilip Kumar Vallabhadas**, Department of Computer Science & Engineering, SRM University-AP.
