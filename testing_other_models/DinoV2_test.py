import os
import torch
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import Dinov2Model, AutoImageProcessor
import torch.nn.functional as F
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ---------------------------- CONFIG ----------------------------
IMAGE_DIR = "./rgb_images"
ANNOTATION_DIR = "./annotations"
MODEL_NAME = "facebook/dinov2-small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "fine_tuned_model.pth"

# ----------------------- LOAD MODEL & PROCESSOR ----------------
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

class TransformerFeatureAdapter(nn.Module):
    def __init__(self, dim, n_heads=4, n_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(x)

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, model_name="facebook/dinov2-small"):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.feature_dim = self.backbone.config.hidden_size
        self.adapter = TransformerFeatureAdapter(self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, 2)

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            features = outputs.last_hidden_state[:, 1:, :]
            features = F.normalize(features, dim=2)

        adapted_features = self.adapter(features)
        B, N, D = adapted_features.shape
        logits = self.classifier(adapted_features.reshape(-1, D)).reshape(B, N, 2)
        return adapted_features, logits
def contrastive_grouping_loss(features, masks, temperature=0.1):
    """
    Args:
        features: [B, N, D] - Feature vectors
        masks: [B, N] - Binary labels (same size as patch grid)
    """
    B, N, D = features.shape
    loss = 0.0
    eps = 1e-6

    for i in range(B):
        f = F.normalize(features[i], dim=1)  # [N, D]
        sim = f @ f.T  # [N, N]
        sim /= temperature

        mask = masks[i]
        pos_mask = (mask[:, None] == mask[None, :]).float()
        neg_mask = 1.0 - pos_mask

        exp_sim = torch.exp(sim)
        pos_sim = (exp_sim * pos_mask).sum(dim=1)
        denom = exp_sim.sum(dim=1) + eps

        loss_i = -torch.log(pos_sim / denom + eps).mean()
        loss += loss_i

    return loss / B

def load_model(fine_tuned=False):
    model = DINOv2SegmentationModel(MODEL_NAME).to(DEVICE)
    if fine_tuned and os.path.exists(MODEL_PATH):
        print("\nLoading fine-tuned model...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
    else:
        print("\nLoading default pretrained model...")
    return model

# -------------------------- HELPERS ----------------------------
def load_masks(hdf5_path):
    masks = []
    with h5py.File(hdf5_path, 'r') as f:
        if 'segments' not in f or len(f['segments']) == 0:
            raise ValueError("No 'segments' found in annotation.")
        for key in f['segments']:
            seg = f[f'segments/{key}'][:]
            if seg.shape and seg.any():
                binary = (seg > 0).astype(np.uint8)
                masks.append(binary)
    if not masks:
        raise ValueError("No valid segments found.")
    return masks

def resize_mask(mask, target_shape):
    from skimage.transform import resize
    resized = resize(mask, target_shape, order=0, preserve_range=True, anti_aliasing=False)
    return (resized > 0.5).astype(np.uint8)

def image_to_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        features, _ = model(inputs['pixel_values'])
    return features.squeeze(0)

def compute_edit_adherence_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def segment_grouping_f1(pred, gts):
    TP, FP, FN = 0, 0, 0
    for gt in gts:
        resized = resize_mask(gt, pred.shape)
        match = np.logical_and(pred, resized).sum() / np.logical_or(pred, resized).sum()
        if match > 0.5:
            TP += 1
        else:
            FN += 1
    FP = max(0, len(np.unique(pred)) - 1 - TP)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)

def split_dataset(image_files, train_ratio=0.8):
    np.random.shuffle(image_files)
    split = int(len(image_files) * train_ratio)
    return image_files[:split], image_files[split:]

# ---------------------  TRAIN FUNCTION ---------------------
def train_model(train_files):
    print("\nStarting training...")
    model.train()
    trainable_params = list(model.adapter.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        print(f"\n[Epoch {epoch + 1}]")
        total_loss, num_batches = 0.0, 0

        for img_file in tqdm(train_files, desc="Training"):
            base_id = img_file.replace("_rgb.png", "")
            image_path = os.path.join(IMAGE_DIR, img_file)
            hdf5_path = os.path.join(ANNOTATION_DIR, f"{base_id}_annotation.hdf5")
            if not os.path.exists(hdf5_path):
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                masks = load_masks(hdf5_path)

                optimizer.zero_grad()
                features, logits = model(inputs['pixel_values'])
                N = features.shape[1]
                patch_dim = int(np.sqrt(N))

                total_image_loss, full_mask = 0.0, np.zeros((patch_dim, patch_dim), dtype=np.int32)
                label_id = 1

                for mask in masks:
                    resized = resize_mask(mask, (patch_dim, patch_dim))
                    full_mask[resized == 1] = label_id
                    label_id += 1

                labels = torch.tensor(full_mask.flatten(), dtype=torch.long, device=DEVICE)
                segment_logits = logits.reshape(-1, 2)
                ce_loss = loss_fn(segment_logits, (labels > 0).long())

                group_loss = contrastive_grouping_loss(features, labels.view(1, -1))
                total_loss_value = ce_loss + 0.1 * group_loss
                total_loss_value.backward()

                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                total_loss += total_loss_value.item()
                num_batches += 1

            except Exception as e:
                print(f"Error: {e}")

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  ➤ Average Loss: {avg_loss:.4f}")
        scheduler.step()

    print("Training complete!")
    torch.save(model.state_dict(), MODEL_PATH)

# --------------------- ATTENTION MAP VISUALIZATION ---------------------
def visualize_attention_map(image_tensor, feature_map, save_path):
    """
    image_tensor: [1, 3, H, W]
    feature_map: [N, D]
    """
    attn = feature_map.norm(dim=-1).detach().cpu().numpy()  # [N]
    patch_dim = int(np.sqrt(attn.shape[0]))
    attn = attn.reshape(patch_dim, patch_dim)

    attn_resized = Image.fromarray((attn / attn.max() * 255).astype(np.uint8)).resize((256, 256))
    img = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
    heatmap = Image.blend(img.convert("L").convert("RGB"), attn_resized.convert("RGB"), alpha=0.5)
    heatmap.save(save_path)


def grouping_f1_score(pred_mask, gt_mask):
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    return f1_score(gt_flat, pred_flat, average='binary')

def evaluate_model(test_files):
    print("\nStarting evaluation...")
    model.eval()
    ea_scores, f1_scores = [], []

    for img_file in tqdm(test_files, desc="Evaluating"):
        base_id = img_file.replace("_rgb.png", "")
        image_path = os.path.join(IMAGE_DIR, img_file)
        hdf5_path = os.path.join(ANNOTATION_DIR, f"{base_id}_annotation.hdf5")
        if not os.path.exists(hdf5_path):
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            masks = load_masks(hdf5_path)

            with torch.no_grad():
                feats, logits = model(inputs['pixel_values'])
                pred_probs = torch.softmax(logits, dim=-1)[0]
                pred_labels = pred_probs.argmax(dim=-1).cpu().numpy()

            patch_dim = int(np.sqrt(pred_labels.shape[0]))
            pred_mask = pred_labels.reshape(patch_dim, patch_dim)

            for mask in masks:
                gt_mask = resize_mask(mask, (patch_dim, patch_dim))
                ea_score = compute_edit_adherence_iou(pred_mask == 1, gt_mask == 1)
                f1 = grouping_f1_score(pred_mask == 1, gt_mask == 1)

                if ea_score > 0:
                    ea_scores.append(ea_score)
                if f1 > 0:
                    f1_scores.append(f1)

                print(f"EA IoU: {ea_score:.3f} | Grouping F1: {f1:.3f}")

        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")

    print("=" * 60)
    if ea_scores:
        print(f"Mean EA (IoU): {np.mean(ea_scores):.4f}")
    else:
        print("No valid EA scores above 0.")
    if f1_scores:
        print(f"Mean Grouping F1: {np.mean(f1_scores):.4f}")
    else:
        print("No valid Grouping F1 scores above 0.")
    print("=" * 60)

# ---------------------------- MAIN -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dinov2 Training and Evaluation Script")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    args = parser.parse_args()

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith("_rgb.png")]
    train_files, test_files = split_dataset(image_files)
    model = load_model(fine_tuned=False if args.train else True)

    if args.train:
        train_model(train_files)
    evaluate_model(test_files)
