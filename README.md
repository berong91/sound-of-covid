# Sound of COVID-19
--------------

# Author
- The Duy Pham" <dpham22@my.bcit.ca>
- Matthew Harrison" <mharrison62@my.bcit.ca> 



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
    ├── test_model.py
    ├── train_model.py
    └── utils.py

```

- data_config.py: This file contains all the configurations for the pipeline.

| Flag                | Type    | Details                                                                                                                                             |
|---------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| EXPORT_IMAGE        | boolean | to export spectrogram and mel spectrum image                                                                                                        |
| APPLY_MFCC          | boolean | to apply MFCCs on the spectrogram data                                                                                                              |
| PREFIX_INPUT        | string  | the prefix folder on where you place the source Coswara repository                                                                                  |
| EXTRACTED_DATA_PATH | string  | the path where the data from Coswara project is extracted                                                                                           |
| PREFIX_MODEL        | string  | the path where model will be saved                                                                                                                  |
| PREFIX_OUTPUT       | string  | the prefix folder for the exported data. This export data is the Mel Spectrogram data and needs to be export to a folder for subsequent retraining. |
| POSTFIX_MODEL       | string  | the postfix name for model                                                                                                                          |
| BANNED_ID           | list    | some records get corrupted, and so their ID will be placed here                                                                                     |
| BANNED_ID_BY_FEAT   | dict    | some records get corrupted at a specific feature, and they will be filtered out for that particular feature                                         |
| SEED                | string  | seed value for randomize                                                                                                                            |
| index_col           | list    | which column is the label index - aka y                                                                                                             |
| key_col             | list    | which column is the training index - aka X                                                                                                          |

- export_data.py: run once to generate the MFCC data. Note that the Output folder defines under of `data_config`


- train_model.py: Once `export_data.py` has been executed, this script to train our model, using the generated Mel
  Spectrum output
    - The model will then exports into the model folder (PREFIX_MODEL) with postfix name POSTFIX_MODEL


- utils.py: Utilities class.


- test_model.py: to test the efficiency of the model using test data

# Step to install

1. Git clone the repository
2. Either run extract_data.sh, or manually download Coswara_Data and extract the data somewhere
3. Edit `data_config.py` to match the input path and output path.
4. Run `export_data.py` to generate the Mel Spectrum data
5. Run `train_model.py` to start training our first model
6. Run `test_model.py` to test result of the trained model

# Packages

- Everything should be listed under requirements.txt

