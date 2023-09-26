# MMSR (Multi-modality enriched Sequential Recommendation)

This is the PyTorch implementation for MMSR proposed in the paper Adaptive Multi-Modalities Fusion in Sequential Recommendation Systems (https://arxiv.org/pdf/2308.15980v1.pdf), CIKM, 2023.
The model name in the code is VLGraph (Visio-linguistic Graph).

### Preprocess

- **First Step**. Download the raw data from http://jmcauley.ucsd.edu/data/amazon/links.html
- **Second Step**. Create an `"image"` folder under the downloaded datafolder, and scrawl the images, and save them into the `"image"` folder.
- **Third Step**. Check `config/preprocess.yaml` for starting dataset preparation.
- **Fourth Step**. Enter the `preprocess/dataset_name/` folder, run `python image_feature_extractor.py`、`python text_feature_extractor_t5.py`、`python process_dataset.py` step by step

Note that the preprocess provide several optiosn: to generate datasets for collaborative filtering task (dataset_cf) sequential recommendation task (dataset_sr), multimodal sequential recommendation task (dataset_mmsr)

### How to Run

Before running the program, you need to check the model configuration file in `config/model.yaml`, and make sure you are using the correctly preprocessed dataset folder.

Process "python main.py dataset_name", such as,

```python
python main.py beauty
```

The results will be saved into `log/`.


### Others
You can also find an example of our execution log as shown in `running_records.log`, which reports the running records and results of our model on the beauty dataset.
