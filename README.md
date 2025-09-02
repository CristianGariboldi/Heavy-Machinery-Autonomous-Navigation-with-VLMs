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

Make sure to assign the right path of the model in the code [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/data_tools/QA_data_generation.py#L24) and the right PYTHONPATH, CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH, BNB_CUDA_VERSION, image folder and output directory paths [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/data_tools/VLM_converter.sh#L2-L19). Moreover, if you have installed Miniconda instead of Anaconda, replace "anaconda3" with "miniconda3" [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/data_tools/VLM_converter.sh#L16).

## Generate Question-Answer Data

To start generating QA data, make sure to be under the root directory of the project, and run:

```
sh data_tools/VLM_converter.sh
```

With my workstation (2 * NVIDIA RTX 6000 Ada Generation), it takes around 70s/it (with 4-bit quantization, set [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/data_tools/QA_data_generation.py#L34)). Feel free to apply 8-bit quantization instead or remove it completely. With 4-bit quantization, generating QA data for around 2200 images takes around 2 days with my setup.

For your convenience, I already provide the generated QA dataset for all the images, so you don't need to generate QA data by yourself (you can find "_all_frames.json" in "output_data" folder).

To inspect the command distribution of the QA dataset, simply run:

```
python3 data_tools/count_actions.py
```

In order to fine tune a VLM on this json dataset, we need to properly convert it to the right format. To do that, simply run:

```
python3 data_tools/convert2llama.py
```

Also in this case, I already provide this file for your convenience, which you can find it in "output_data" folder, called "llama_format_dataset.json". Before running the script, just make sure to provide the right paths [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/data_tools/convert2llama.py#L61-L67).


We can now move to the next step.


## Fine-tuning

We are going to fine-tune a more lightweight model, specifically vicuna-7b-v1.5, which you can download from [here](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main).

Once downloaded locally, make sure to specify the right paths [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/train_tools/train_lora.sh#L3-L5), [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/train_tools/train_lora.sh#L20-L22), [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/train_tools/train_lora.sh#L24-L25) and [here](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/train_tools/train_lora.sh#L32).

Now, before starting the training, install the required libraries:

```
pip install -r requirements_training.txt
```

Once everything is set up, simply run:

```
sh train_tools/train_lora.sh
```

The training loop using LoRA will start and once finished it will save the fine-tuned model in the output directory.
If you have the computation capabilities to run a full-parameter fine-tuning instead of using LoRA, simply remove [this line](https://github.com/CristianGariboldi/Heavy-Machinery-Autonomous-Navigation-with-VLMs/blob/main/train_tools/train_lora.sh#L26).


## Evaluation

