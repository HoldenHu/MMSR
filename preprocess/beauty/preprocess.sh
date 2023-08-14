#!/bin/bash

echo "> Running image_feature_extractor.py"
python image_feature_extractor.py
echo "> Finished running image_feature_extractor.py"

echo "> Running text_feature_extractor_t5.py"
python text_feature_extractor_t5.py
echo "> Finished running text_feature_extractor_t5.py"

echo "> Running process_dataset.py"
python process_dataset.py
echo "> Finished running process_dataset.py"
