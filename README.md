# Brain Tumor MRI Image Classification

PyTorch project for classifying MRI images into glioma, meningioma, no tumor, and pituitary classes. The trained checkpoints are included for the Streamlit application. Dataset images are kept out of this repository; download the licensed dataset and place it under `Tumour/{train,valid,test}/<class>/` to run the training notebooks.

## Dataset observations

The folder structure has four labels and ready-made train/validation/test splits. Current image counts read by the project loader are train: 1,695, validation: 502, and test: 246. These match the dataset README. Class counts and sample image dimensions are printed in the notebook. Images vary in resolution. The committed training run resized them to 96×96; new runs default to 160×160, and inference reads the size stored in each checkpoint.

Dataset attribution: Labeled MRI Brain Tumor Dataset by Ali Rostami, published through [Roboflow Universe](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset), licensed under CC BY 4.0. See [DATASET_ATTRIBUTION.md](DATASET_ATTRIBUTION.md) for details.

## Train

Install `requirements.txt`, inspect `Dataset_Exploration.ipynb`, then open `Brain_Tumor_MRI_Classification.ipynb` and run the cells. The included CPU runs used 6 epochs for the CNN and 3 for ImageNet-pretrained MobileNetV2 at 96×96; held-out test accuracies were 75.6% and 93.9%, respectively. These are single-split prototype results and may not generalize beyond this dataset. This trains a custom CNN and MobileNetV2, evaluates each on the existing held-out test split, and saves `.pt` checkpoints and JSON metrics under `models/`. For a single model from the command line:

```sh
python train.py --model cnn --epochs 8
python train.py --model mobilenet_v2 --epochs 8
```

MobileNetV2 requests torchvision's ImageNet weights. If they cannot be fetched, the code reports that fact and continues without pretrained weights; check `models/mobilenet_v2_results.json` before describing it as transfer learning. This project has CPU-only training available in the current environment; adjust epochs/batch size for available hardware.

## Run the app

```sh
streamlit run app.py
```

Choose a trained model checkpoint in the sidebar and upload a JPG, JPEG, PNG, or BMP MRI image.

## Medical use

This is an educational prototype, not a diagnostic product. Predictions can be wrong and must not guide patient care. The dataset's provenance and labeling are not independently validated here.
