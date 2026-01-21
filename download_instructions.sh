#!/bin/bash
# Run these commands to download missing datasets (requires kaggle CLI and accepted competitions)
kaggle datasets download -d ashokpant/drought-detection -p drought --unzip
kaggle datasets download -d pratik2901/plantdoc-dataset -p plantdoc --unzip
Please obtain IP102 from official source or mirrors and place under ip102/
kaggle datasets download -d kmader/plant-seedlings-classification -p plant_seedlings --unzip
kaggle competitions download -c plant-pathology-2020-fgvc7 -p plant_pathology
