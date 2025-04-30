# SAM Segment Manual Dataset Collection Tool

This tool allows you to manually collect segment annotations using [Meta's Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) with a Gradio interface. Upload an image, click to add points as prompts, run SAM to generate segmentation masks, and save selected masks along with their associated point prompts.

## 🧰 Features

- Click-based point prompt interface
- Run SAM to generate up to 3 segmentation masks
- Save masks with associated points
- Finalize and store annotations in HDF5 format
- Visual history of all annotated sessions

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/tahmidjamal12/sam.git
cd sam
```
### 2. Create a Conda Environment
Verify you have Anaconda installed then run the following command

```
conda create -f environment.yml
conda activate sam_gradio
```

### 3. Download the SAM ckpt file
When running the following command for the first time it will install ckpt.

#### If CUDA is installed
```
python SAM_dataset_collection_gradio.py --checkpoint 'path_to_ckpt' --rgb_save_path "path_to_rgb_save" --annotation_save_path "path_to_annotation_save"
```
#### If not installed
```
python SAM_dataset_collection_gradio.py --checkpoint 'path_to_ckpt' --rgb_save_path "path_to_rgb_save" --annotation_save_path "path_to_annotation_save" --device "cpu"
```

### 4. Open the website
In the terminal it will host a gradio app and you click on the url to open it in your local browser.
