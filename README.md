# Lung Region Extractor for Pediatric Chest X-ray Images
Multi-view deep learning-based solution that extracts lung and mediastinal regions of interest from pediatric chest X-ray images where key Tuberculosis findings may be present.

[arXiv](https://doi.org/10.48550/arXiv.2301.13786)  |  [Paper](https://doi.org/10.1117/12.2652626)

## Introduction
Tuberculosis (TB) is still considered a leading cause of death and a substantial threat to global child health. Both TB infection and disease are curable using antibiotics. However, most children who die of TB are never diagnosed or treated. In clinical practice, experienced physicians assess TB by examining chest X-rays (CXR). Pediatric CXR has specific challenges compared to adult CXR, which makes TB diagnosis in children more difficult. Computer-aided diagnosis systems supported by Artificial Intelligence have shown performance comparable to experienced radiologist TB readings, which could ease mass TB screening and reduce clinical burden. We propose a multi-view deep learning-based solution which, by following a proposed template, aims to automatically regionalize and extract lung and mediastinal regions of interest from pediatric CXR images where key TB findings may be present. Experimental results have shown accurate region extraction, which can be used for further analysis to confirm TB finding presence and severity assessment.

<p align="center">
<img src="img/pipeline_icip_paper_v4.drawio.png" alt="Lung Region Extraction for Pediatric Chest X-ray Images - Pipeline" height="400" title="Lung Region Extraction for Pediatric Chest X-ray Images - Pipeline">
</p>

<p align="center">
<img src="img/montage_labels_final.jpg" alt="Lung Region Extraction for Pediatric Chest X-ray Images - Results" height="250" title="Lung Region Extraction for Pediatric Chest X-ray Images - Results">
</p>

## Table of Contents

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
    - [Minimal Working Example (MWE)](#minimal-working-example-mwe)
    - [Useful Tips and Considerations](#useful-tips-and-considerations)
- [References](#references)
- [How to cite](#how-to-cite)
- [License](#license)

## Installation

We highly recommend creating a virtual environment for this task. Please follow the steps:

1. Run `install_environment.sh`:

    ```
    bash install_environment.sh
    ```
    
    If using other paths or directories for installation, please modify `install_environment.sh` accordingly.

    **Note: Check that you're located in the cloned directory and not in any subfolder before executing this**

2. Copy nnUNetTrainerV2_50epochs.py to nnUNet's trainers directory. If using default paths, inside the cloned directory run:

    ```bash
    cp ./src/nnUNetTrainerV2_50epochs.py ~/nnUNet/nnunet/training/network_training/nnUNetTrainerV2_50epochs.py
    ```
    **Note: Check that you're located in the cloned directory and not in any subfolder before executing this**
3. (Optional) Download pre-trained models from [here](docs/models.md). All weights except those of nnU-Net are already downloaded when cloning the repo. Please refer to the previous link to download nnU-Net weights and place them in a folder called nnunet_weights in the code directory (./nnunet_weights).
4. (Optional) Install the nnU-Net 2D models (.zip) by entering the following commands:

    ```bash
    nnUNet_install_pretrained_model_from_zip ./nnunet_weights/pTB_nnunet_model_AP.zip
    nnUNet_install_pretrained_model_from_zip ./nnunet_weights/pTB_nnunet_model_LAT.zip
    ```

**Important: Please restart your bash/terminal after these steps, otherwise the environment variables may not be loaded properly.**

## Usage

Command for running the process:

`python process.py --csv CSV_FILE --seg_model MODEL --output_folder OUT_FOLDER --fformat FILE_FORMAT`

where:
- `CSV_FILE` refers to a CSV file with the data inputted to the system (see example in dataset.csv).
- `MODEL` selects which model is used for the semantic segmentation process. It can be either `nnunet` [1], `medt` or `gatedaxialunet` [2] (see references for further information).
- `OUT_FOLDER` is the folder where the results are saved.
- `FILE_FORMAT` refers to the format of the input images. It can be either `jpg`,`png` or `nifti`.

By default, preprocessing with CLAHE is applied to the input images. If CLAHE is not desired in the preprocessing step, add the flag `--no_clahe` at the end of the previous command.

All results are saved in the output folder (`OUT_FOLDER`) specified in the abovementioned command. The resulting regions and crops are saved in `regions` subfolder.

### Minimal Working Example (MWE)

The following command executes a minimal working example (MWE). Please check that everything works as expected.

    python process.py --csv dataset.csv --seg_model nnunet --output_folder ./RESULTS/mwe --fformat jpg

### Useful Tips and Considerations

- If using MedT or GatedAxialUNet for the segmentation process, only use one GPU for inference (Multi-GPU for inference is not recommended in these cases and may deliver wrong results). To do so, use `CUDA_VISIBLE_DEVICES=0,1,...` at the beggining of the command. Following the MWE, if we want to use GPU #0:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python process.py --csv dataset.csv --seg_model nnunet --output_folder ./RESULTS/mwe --fformat jpg
    ```

- In the CSV, use the same column names as in the sample. `PatientID` string should be part of AP/LAT image filenames. Example: `ITA2-0326` -> `TB0_ITA2-0326-AP-20130116.jpg`

- DO NOT name the input files with:
    - More than one underscore.
    - Only numbers.
    
    Otherwise, the process may raise a exception. Suggested file naming: <COHORT IDENTIFIER>_<CASE_IDENTIFIER>.<FORMAT>. Example: COH_001.jpg

## References

[1] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” Nat. Methods 2020 182, vol. 18, no. 2, pp. 203–211, Dec. 2020, doi: 10.1038/s41592-020-01008-z.

[2] J. Maria, J. Valanarasu, P. Oza, I. Hacihaliloglu, and V. M. Patel, “Medical transformer: Gated axial-attention for medical image segmentation,” arxiv.org, Accessed: Dec. 16, 2021. [Online]. Available: https://arxiv.org/abs/2102.10662

## How to cite

Please cite us if you are using our code and/or weights!

```
Daniel Capellán-Martín, Juan J. Gómez-Valverde, Ramon Sánchez-Jacob, David Bermejo-Peláez, Lara García-Delgado, Elisa López-Varela, Maria J. Ledesma-Carbayo, "Deep learning-based lung segmentation and automatic regional template in chest x-ray images for pediatric tuberculosis," Proc. SPIE 12465, Medical Imaging 2023: Computer-Aided Diagnosis, 124651W (7 April 2023); https://doi.org/10.1117/12.2652626
```

BibTeX format:

```
@inproceedings{10.1117/12.2652626,
author = {Daniel Capell{\'a}n-Mart{\'i}n and Juan J. G{\'o}mez-Valverde and Ramon S{\'a}nchez-Jacob and David Bermejo-Pel{\'a}ez and Lara Garc{\'i}a-Delgado and Elisa L{\'o}pez-Varela and Maria J. Ledesma-Carbayo},
title = {{Deep learning-based lung segmentation and automatic regional template in chest x-ray images for pediatric tuberculosis}},
volume = {12465},
booktitle = {Medical Imaging 2023: Computer-Aided Diagnosis},
editor = {Khan M. Iftekharuddin and Weijie Chen},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {124651W},
keywords = {Tuberculosis, Lung, Pediatric chest X-ray, Medical imaging, Computer vision, Deep learning, Semantic segmentation},
year = {2023},
doi = {10.1117/12.2652626},
URL = {https://doi.org/10.1117/12.2652626}
}
```

Note: This work has been accepted at the SPIE Medical Imaging 2023, Image Processing conference, 19-23 Feb 2023, San Diego, California, United States.
    
## License
    
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

Images in `sample_images/` must not be used without the consent of the authors. They serve only as a minimal working example (MWE) for the code.

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
