# Sound of COVID-19
--------------

# Extracting data

You can pull the data directly from https://github.com/iiscleap/Coswara-Data, or run the bash script `extract_data.sh`
under the data folder.

This file will auto pull the sound repository, then extract all the sounds into the data folder.

# Data pipeline

Basic pipeline:

```
.
├── README.md
├── data
│   ├── Coswara_Data
│   ├── extract_data.sh
│   └── extracted
├── requirement.txt
└── src
    ├── data_config.py
    ├── export_data.py
    ├── train_model.py
    └── utils.py

```

- data_config.py: This file contains all the configurations for the pipeline.
  
| Flag                | Type    | Details                                                                                                                                             |
|---------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| EXPORT_IMAGE        | boolean | to export spectrogram and mel spectrum image                                                                                                        |
| PREFIX_INPUT        | string  | The prefix folder on where you place the source Coswara repository                                                                                  |
| EXTRACTED_DATA_PATH | string  | the path where we extract the data from Coswara project                                                                                             |
| PREFIX_OUTPUT       | string  | The prefix folder for the exported data. This export data is the Mel Spectrogram data and needs to be export to a folder for subsequent retraining. |
| BANNED_ID           | list    | Some records get corrupted, and so their ID will be placed here                                                                                     |
| index_col           | list    | Which column is the label index - aka y                                                                                                             |
| key_col             | list    | which column is the training index - aka X                                                                                                          |
  

- export_data.py: run once to generate the MEL spectrogram. Note that the Output folder defines under of `data_config`


- train_model.py: After `export_data` has been executed, run this script to train our model, using the generated Mel
  Spectrum output.
    - The model will then exports into the same folder with timestamp.


- utils.py: Utilities class.

# Step to install

1. Git clone the repository
2. Either run extract_data.sh, or manually download Coswara_Data and extract the data somewhere
3. Edit `data_config.py` to match the input path and output path.
4. Run `export_data.py` to generate the Mel Spectrum data
5. Run `train_model.py` to start training our first model
6. ???

# Packages

- Everything should be listed under requirements.txt

