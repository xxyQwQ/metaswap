<h1 align="center">
MetaSwap: Towards Elegant One-Shot Face Swapping Framework
</h1>
<p align="center">
    Project of AI2612 Machine Learning Project, 2023 Fall, SJTU
    <br />
    <a href="https://github.com/Ark-ike"><strong>Yi Ai</strong></a>
    &nbsp;
    <a href="https://github.com/xxyQwQ"><strong>Xiangyuan Xue</strong></a>
    &nbsp;
    <a href="https://github.com/YsmmsY"><strong>Shengmin Yang</strong></a>
    <br />
</p>

The framework is designed based on [faceshifter](https://arxiv.org/pdf/1912.13457.pdf) and [simswap](https://arxiv.org/pdf/2106.06340.pdf). The face swapping model mainly supports 224x224 resolution.

## Requirements

To ensure the code runs correctly, following packages are required:

* `python`
* `hydra`
* `opencv`
* `pytorch`

You can install them following the instructions below.

* Create a new conda environment and activate it:
  
    ```bash
    conda create -n metaswap python=3.10
    conda activate metaswap
    ```

* Install [pytorch](https://pytorch.org/get-started/previous-versions/) with appropriate CUDA version, e.g.
  
    ```bash
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

* Install `hydra` and `opencv`:
  
    ```bash
    pip install hydra-core
    pip install opencv-python
    ```

Latest version is recommended for all the packages, but make sure that your CUDA version is compatible with your `pytorch`.

## Preparation

In this project, [insightface](https://github.com/TreB1eN/InsightFace_Pytorch) is required for face detection and alignment. Relevant files are already included in the `facial` directory. You should download the pretrained weights [here](https://drive.google.com/open?id=15nZSJ2bAT3m-iCBqP3N_9gld5_EGv4kp) and save it as `facial/weight.pth`.

If you have installed `gdown`, this step can be done by running the following command:
```bash
gdown 15nZSJ2bAT3m-iCBqP3N_9gld5_EGv4kp -O facial/weight.pth
```

## Training

To train the model, `vggface2` dataset is recommended. You can download it [here](https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view). The directory should contain multiple subdirectories, each of which contains face images with the same identity.

Then modify the configuration file in the `config` directory, where `dataset_path` must be correctly set as the path to your dataset. You can also modify the hyperparameters or create a new configuration file as you like, but remember to modify the `hydra` arguments in `training.py` accordingly.

Here we provide a template configuration file `config/training.yaml`. The batch size is set to 16 by default, which requires at least 16GB GPU memory. Feel free to reduce the batch size if you do not have enough GPU memory.

Run the following command to train the model:

```bash
python training.py
```

## Inference

The template configuration file `config/inference.yaml` contains necessary arguments for inference. You should correctly set `model_path` as the path to your generator model. Then `source_image_path` and `target_image_path` should be the paths to your source image and target image respectively. The output image will be saved in the `checkpoint` directory by default, but you can specify it as you like.

You can download the pretrained model [here](https://github.com/xxyQwQ/metaswap), but training the model by yourself provides better customization.

Run the following command to perform inference:

```bash
python inference.py
```

Since `hydra` allows you to override the arguments in the configuration file in the command line, you can also run the following command as an alternative:

```bash
python inference.py parameter.source_image_path=sample/image_1.jpg parameter.target_image_path=sample/image_2.jpg
```
