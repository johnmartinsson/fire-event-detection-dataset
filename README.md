# Fire event detection dataset
This repository contains the instructions on how to download and prepare the fire event detection dataset, and how to to download and evaluate the baseline model, as well as how to train the baseline model on this dataset.

Follow the instructions in the order they are presented below to recreate the train/validation/test split of the paper, and to re-produce the main results of the paper:

- figure 5,
- figure 6, and
- table 5.

## Clone repository and setup environment

    git clone https://github.com/anonymous7483/fire-event-detection-dataset.git
    cd fire-event-detection-dataset
    pip install -r requirements.txt

## Download prepared data and model weights
This will download the exact datasplit and model weights used in the paper

    wget https://www.dropbox.com/s/rvwwmt2e90z41cy/experiments.zip
    unzip experiments.zip
    wget https://www.dropbox.com/s/0xkknctvt84nszl/dataset_spruce_oak_pmma_pur_chipboard_sr_32000.hdf5

## Produce figures and tables

    python produce_figures_and_tables.py "cpu" # or "cuda" if available
    
This will print table 5 in the console and save "figure_5_a.pdf", "figure_5_b.pdf", and "figure_6.pdf" to disk in the working directory.
    
# Training model
This does not guarantee the exact same results as in the paper, but they should be close.

## Download and prepare data (optional)
The training dataset can be reproduced by downloading the original sound recordings and extracting the segments using the provided script. If you downloaded the prepared dataset already this step can be skipped. The data can be downloaded from the following Dropbox link:

    wget https://www.dropbox.com/s/7tkeh7wwuofe9zl/data.zip
    unzip data.zip
    
There should now be a folder named "wav" with the audio source files in the working directory.

To prepare the dataset run

    python prepare_data.py

This will create a HDF5 file with the prepared dataset, by default using the path "./wav" for the audio source files. This can take a while. We opted for releasing the data with full sampling rate, and as a result this step takes some more time. Most of the time is spent on resampling the audio files to 32,000 Hz.

## Train model
This will train a model, by default the training output is stored in the folder: "experiments/baseline", with the configuration used to find the model presented in the paper.

    python baseline.py "train_model" "cuda" # or "cpu" if available

## Evaluate model
This will evaluate the trained model on the test dataset.

    python baseline.py "evaluate_model" "cuda" # or "cpu" if available
