# Open-set Mosquito Classification on Crowdsourced data using two-stage system with OpenMax

This is the code repository for the paper "Mosquito Identification with Open-set Recognition Using Crowdsourced Images". This project is funded by PAN-ASEAN Coalition for Epidemic and Outbreak Preparedness (PACE-UP). It is largely inspired from the 7th place solution for the MosquitoAlert Challenge 2023 by HCA97. Link to the original [here](https://github.com/HCA97/Mosquito-Classifiction/tree/main).


## Train CLIP Classifier

1. **Install Datasets**
   - Download the competition dataset from [here](https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023/dataset_files) and unzip it to a folder named `data_round_2` (the annotations files are included).
   - Install GBIF [dataset](https://www.kaggle.com/datasets/lekoup/gbif-residual-cropped) and unzip it to `gbif-cropped` folder (the annotations file is included).

2. **Install Dependencies**
   - Use the following command to install the necessary dependencies: `pip install -r requirements.txt`.

3. **Run the Classifier**
   - Navigate to the `experiments` directory and run `train_clip.ipynb`.

## How to Train YOLOv8-s Model

1. **Install Competition Dataset**
   - Download the competition dataset from [here](https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023/dataset_files) and unzip it to a folder named `data_round_2`.

2. **Install Dependencies**
   - Use the following command to install the necessary dependencies: `pip install -r requirements.txt`.

3. **Prepare YOLO Dataset**
   - Navigate to the `experiments/yolo` directory and run the script: `python convert_mosquito_to_yolo.py`.

4. **Start Training**
   - Execute the command: `python yolo_training.py`.

## YOLO + CLIP with OpenMax
   - Navigate to `experiments` folder for experimentations.
   - Insert weights path of YOLO and CLIP into and run `yoloclip.ipynb` to run pipeline with OpenMax.
   - To compare YOLO with YOLO + CLIP pipeline, use `yolo_vs_yoloClip.ipynb`.


## Annotation Files

### data_round_2

- `phase2_train_v0_cleaned.csv` was created using `owl-vit`. You can refer to `experiments/cleaning_annotations.ipynb` for details.
- `phase2_train_v0_cleaned_yolo_best_annotations.csv` uses `phase2_train_v0_cleaned.csv` along with YOLOv8-s model annotations. Refer to `extra_data/annotate_images_yolo.py` for more information.
- `best_model_val_data_yolo_annotations.csv` and `best_model_train_data_yolo_annotations.csv` are train/validation splits of `phase2_train_v0_cleaned_yolo_best_annotations.csv`.

### gbif-cropped

- `inaturalist.csv` contains annotations for lux's dataset. Since the images are already cropped, we used the entire image as the bounding box.
- `ma_lux1.csv` is a combination of the `best_model_val_data_yolo_annotations.csv`, `best_model_train_data_yolo_annotations.csv`, and the GBIF `inaturalist.csv` annotation.

## Guided Grad-CAM

