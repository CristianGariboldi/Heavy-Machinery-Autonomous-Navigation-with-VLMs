# Heavy-Machinery-Autonomous-Navigation-with-VLMs
This repository's aim is to enhance autonomous navigation and actions of heavy machines in construction zones leveraging VLMs.

## Setup

```
git clone https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs.git
conda create -n sensmore python=3.10 -y
conda activate sensmore
cd Heavy-Machinery-Autonomous-Navigation-with-VLMs
pip install -r requirements.txt
```

Now, install PyTorch and TorchVision:
```
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Before running the code, move into **"data"** folder. There are 2 subfolders, namely:

1) CLIP
2) llava34b

In **"CLIP"** folder, you should download the OPENAI [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) model from HuggingFace.

Instead, in **"llava34b"** folder, you should download the [LLaVA-v1.6-34b](https://huggingface.co/liuhaotian/llava-v1.6-34b/tree/main) model from HuggingFace.

Make sure to assign the right path of the model in the code [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/data_tools/QA_data_generation.py#L105) (or [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/data_tools/QA_data_generation.py#L76)) and the right PYTHONPATH, data and output directory paths [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/data_tools/VLM_converter.sh#L3-L11). Moreover, if you have installed Anaconda instead of Miniconda, replace "minicoda3" with "anaconda3" [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/data_tools/VLM_converter.sh#L8).