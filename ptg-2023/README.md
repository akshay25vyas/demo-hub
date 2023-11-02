# PTG 2023 Demo

## Setup
```sh
# clone
git clone --recursive https://github.com/IRVLUTD/demo-hub.git

conda create -n ptg23-demo python=3.9
conda activate ptg23-demo

# Install the according versions of torch and torchvision
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

cd utils/submodules/segment-anything; pip install -e .; cd ..

export CUDA_HOME=/usr/local/cuda # adjust based on your CUDA installation
cd utils/submodules/GroundingDINO; pip install -e .; cd ..

#check if pytorch is installed with cuda support if you have gpu
python -c "import torch; print(torch.cuda.is_available())" 
```

## Download Pretrained Checkpoints
```sh
mkdir -p pretrained_checkpoints
cd pretrained_checkpoints
mkdir sam gdino
cd sam; wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth; wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt; cd ..
cd gdino; wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth; cd ..
cd ../..
```

## Run Obj detection using GroudningDINO and SAM/MobileSAM
```sh
python run_gdino_sam.py --dataset_path </home/user/Datasets/ptg-november23demo/Object_Detection/>
```

## Data-Preprocessing: Crop [For Proto-CLIP]
```sh
python ptg-2023/utils/crop_hololens_data.py \
--xlsx_file </home/user/Datasets/ptg-november23demo/Object_Detection/PTG_Demo_2023_DataCollection.xlsx> \
--dataset_path </home/user/Datasets/ptg-november23demo/Object_Detection> \
--max_workers 10 # for parallelization
```

#### Steps
- Reads the xlsx file and creates a classname->sequence-folder dictionary mapping
  - As of now 
    - only one sequence is considered even if there are multiple sequences
    - class with empty sequence is omitted
- Then read the color-mask pairs in each sequence-folder
  - Randomly sample <sample_threshold> #frames
  - Crop and save to cropped_objs folder with classnames as the sub-folders

## Compute Image prototypes [For Proto-CLIP]
```sh
python ptg-2023/utils/compute_prototpyes.py --dataset_path </home/user/Datasets/ptg-november23demo/Object_Detection>
```

#### Steps
- For Proto-CLIP make sure the support samples are organized as
```
├── SupportDATA
    ├── classname-1/imgs
    ├── classname-2/imgs
    ├── classname-3/imgs
    ├── classname-4/imgs
    ├── classname-5/imgs
    .
    .
    ├── classname-n/imgs
```
- Cache the training free proto-clip v+l memory banks using the cropped images from each class
- Test on the query images

## Run Obj detection using GroudningDINO + SAM/MobileSAM + Proto-CLIP
```sh
cd ptg-2023;
python run_proto_clip.py \
--dataset_path </home/user/Datasets/ptg-november23demo/Object_Detection/> \
--box_threshold 0.5 --text_threshold 0.5 \
--alpha 0.5 --beta 15 --pct 0.2
```