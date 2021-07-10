# Sound of Covid
--------------

# Extracting data
You can pull the data directly from https://github.com/iiscleap/Coswara-Data, or run the bash script `extract_data.sh` under the data folder. 

This file will auto pull the sound repository, then extract all the sounds into the data folder.

# Data pipeline
I have set up a very basic pipeline.

```
src/
├── data_config.py
├── export_data.py
├── train_model.py
└── utils.py
```

- data_config.py: This file contains all the configurations for the pipeline.
  - PREFIX_INPUT: The prefix folder on where you place the sound repository. Replace with full path if needed
  - PREFIX_OUTPUT: The prefix folder for the exported data. This export data is the Mel Spectrogram data and needs to be export to a folder for subsequent retraining.
  - BANNED_ID: Some records are corrupted, and so their ID will be placed here
  - index_col: Which column is the label index - aka y
  - key_col = which column is the training index - aka X
  
- export_data.py: run once to generate the MEL spectrogram. Note that the Output folder is defined under of data_config

- train_model.py: After `export_data` has been executed, run this script to train our model, using the generated Mel Spectrum output.

- utils.py: Utilities class.
