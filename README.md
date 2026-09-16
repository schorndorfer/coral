# CORAL

This repository contains the code and guidelines to benchmark open source models on the CORAL dataset released as a part of the paper 
[CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference](https://ai.nejm.org/doi/full/10.1056/AIdbp2300110).
If you are mainly interested in replicating the results of the paper, please use code in [this](https://github.com/MadhumitaSushil/OncLLMExtraction) repository instead.

Dataset of 20 breast cancer and 20 pancreatic cancer progress notes were annotated comprehensively by medical experts to encode information of clinical interest from clinical notes. The annotations encompassed 9028 entities, 9986 modifiers, and 5312 relationships. This dataset was further used to benchmark zero-shot, oncology-specific relational information extraction capability of closed and open source LLMs. The dataset can be downloaded [here](https://physionet.org/content/curated-oncology-reports/1.0/) for non-commercial research purposes. Please note that if the dataset is being used to evaluate any proprietary models for example OpenAI models or Google models, it needs to be done within a secure, HIPAA-compliant framework such that no data is ever permanently transferred to or monitored by the underlying company. Azure OpenAI studio with all data transfer, monitoring, and filtering turned off may be a compatible solution.

If you use the dataset or any parts of this code for your research, please cite the corresponding paper, the dataset, and PhysioNet as the following:

CORAL paper and code bibtex citation:
```
@article{doi:10.1056/AIdbp2300110,
author = {Madhumita Sushil  and Vanessa E. Kennedy  and Divneet Mandair  and Brenda Y. Miao  and Travis Zack  and Atul J. Butte },
title = {CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference},
journal = {NEJM AI},
volume = {0},
number = {0},
pages = {AIdbp2300110},
year = {},
doi = {10.1056/AIdbp2300110},

URL = {https://ai.nejm.org/doi/abs/10.1056/AIdbp2300110},
eprint = {https://ai.nejm.org/doi/pdf/10.1056/AIdbp2300110}
,
    abstract = { We curated and assessed a comprehensive dataset of breast and pancreatic cancer oncology progress notes for benchmarking oncology information extraction abilities of large language models. This new dataset has been made available to further advance large language models research in oncology. }
}

```

Dataset citation:

```
Sushil, M., Kennedy, V., Mandair, D., Miao, B., Zack, T., & Butte, A. (2024). CORAL: expert-Curated medical Oncology Reports to Advance Language model inference (version 1.0). PhysioNet. https://doi.org/10.13026/v69y-xa45.
```

PhysioNet citation:

```
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
```

### LICENSE
The code and annotation schema is shared under Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)
Further details can be found on [this](https://creativecommons.org/licenses/by-nc-sa/4.0/) page. 
Additionally, the dataset derived from this schema is shared under the PhysioNet Credentialed Health Data License 1.5.0, which is intended to be used only within non-commercial, sharealike setups similar to the CC BY-NC-SA license.

## Running on Apple Silicon

Run the following commands from the repository root. Python 3.12 is recommended for broad PyTorch and Transformers compatibility.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch transformers accelerate pandas numpy evaluate rouge-score absl-py scikit-learn nltk sentencepiece huggingface-hub streamlit
```

Confirm that PyTorch can use the Mac GPU:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

The result should be `True`. CORAL automatically prefers CUDA, then Apple Metal Performance Shaders (MPS), and finally CPU. MPS uses half-precision model weights and does not enable CUDA-only 8-bit loading.

The annotated dataset directory must contain matching BRAT `.txt` and `.ann` files. Create the inference data with:

```bash
mkdir -p data output
python -m coral.dataprocessor.create_inference_data \
  -annot_data_dir /absolute/path/to/annotated/dataset \
  -fdata coral_inference.csv \
  -dir_data ./data
```

Models are loaded from local files by default. A 1.5B to 3B instruct model is a practical starting point for a Mac with 16 GB of unified memory:

```bash
hf download Qwen/Qwen2.5-3B-Instruct \
  --local-dir ./models/qwen-2.5-3b-instruct
```

Run inference with batch size 1 to limit memory usage. `PYTORCH_ENABLE_MPS_FALLBACK=1` allows operations unsupported by MPS to fall back to CPU:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -m coral.benchmarking.open_source_benchmarking \
  -fdata coral_inference.csv \
  -fout qwen_outputs.csv \
  -dir_data ./data \
  -dir_out ./output \
  -model_name_or_path ./models/qwen-2.5-3b-instruct \
  -batch_size 1
```

Evaluate the saved model responses with:

```bash
python -m coral.benchmarking.evaluate_model \
  -fdata coral_inference.csv \
  -fout qwen_outputs.csv \
  -dir_data ./data \
  -dir_out ./output
```

Inference appends to an existing output CSV, so use a new output filename when starting a fresh run.

## Exploring annotations locally

The annotation viewer reads a local BRAT dataset directory only; it does not upload notes or call external services.

```bash
CORAL_DATA_DIR=/absolute/path/to/coral/annotated \
streamlit run streamlit_app.py
```
