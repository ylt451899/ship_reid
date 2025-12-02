# ship_reid_all_in_one.py
import os
import glob
import random
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models, transforms
import torch.nn.functional as F
import faiss
from tqdm import tqdm
import shutil # 🌟 新增：用於跨平台檔案複製
# -----------------------------
# 配置區
# -----------------------------
DATA_ROOT = "./ReidDataset/ReidDataset/Original_pictures"
OUTPUT_DIR = "./ship_feature"
TRAIN_RATIO = 0.8
IMG_SIZE = (256, 256)
BATCH_P = 16
BATCH_K = 4
MARGIN = 0.3
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 1.2  # L2 distance 判定同艘船

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "gallery"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "query"), exist_ok=True)

# # -----------------------------
# # Step1: 整理資料
# # -----------------------------
# def prepare_data(data_root, train_ratio=0.8):
#     all_classes = sorted(os.listdir(data_root))
#     id_map = {}
#     id_counter = 0
#     for cls in all_classes:
#         images = glob.glob(os.path.join(data_root, cls, "*.jpg"))
#         random.shuffle(images)
#         n_train = int(len(images) * train_ratio)
#         train_imgs = images[:n_train]
#         val_imgs = images[n_train:]

#         # copy train
#         for img_path in train_imgs:
#             cls_name = str(id_counter)
#             save_dir = os.path.join(OUTPUT_DIR, "train", cls_name)
#             os.makedirs(save_dir, exist_ok=True)
#             os.system(f'cp "{img_path}" "{save_dir}/"')
#         # copy val
#         for img_path in val_imgs:
#             cls_name = str(id_counter)
#             save_dir = os.path.join(OUTPUT_DIR, "val", cls_name)
#             os.makedirs(save_dir, exist_ok=True)
#             os.system(f'cp "{img_path}" "{save_dir}/"')

#         id_map[cls] = id_counter
#         id_counter += 1
#     # 可選將 val 部分作 query/gallery
#     for cls in all_classes:
#         val_dir = os.path.join(OUTPUT_DIR, "val", str(id_map[cls]))
#         imgs = glob.glob(os.path.join(val_dir, "*.jpg"))
#         random.shuffle(imgs)
#         mid = len(imgs)//2
#         query_imgs = imgs[:mid]
#         gallery_imgs = imgs[mid:]
#         for img in query_imgs:
#             os.makedirs(os.path.join(OUTPUT_DIR, "query"), exist_ok=True)
#             os.system(f'cp "{img}" "{OUTPUT_DIR}/query/"')
#         for img in gallery_imgs:
#             os.makedirs(os.path.join(OUTPUT_DIR, "gallery"), exist_ok=True)
#             os.system(f'cp "{img}" "{OUTPUT_DIR}/gallery/"')
#     return id_map

# -----------------------------
# Step1: 整理資料 (修正 os.system 的部分)
# -----------------------------
def prepare_data(data_root, train_ratio=0.8):
    id_map = {}
    id_counter = 0
    
    # Get all category directories (e.g., "1 High-speed ship", "2 Warship")
    category_dirs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

    for category in category_dirs:
        category_path = os.path.join(data_root, category)
        
        # Get all class directories within the category (e.g., "0001 VICTORIA CLIPPER V")
        class_dirs = sorted([d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))])

        for cls in class_dirs:
            cls_path = os.path.join(category_path, cls)
            
            # Find all images (jpg, png, jpeg) in the class directory
            images = glob.glob(os.path.join(cls_path, "*.jpg")) + \
                     glob.glob(os.path.join(cls_path, "*.png")) + \
                     glob.glob(os.path.join(cls_path, "*.jpeg"))

            if not images:
                continue

            random.shuffle(images)
            n_train = int(len(images) * train_ratio)
            train_imgs = images[:n_train]
            val_imgs = images[n_train:]

            # Assign a unique ID to this class
            class_id_str = str(id_counter)
            id_map[cls] = id_counter

            # Copy training images
            train_save_dir = os.path.join(OUTPUT_DIR, "train", class_id_str)
            os.makedirs(train_save_dir, exist_ok=True)
            for img_path in train_imgs:
                shutil.copy(img_path, train_save_dir)
                
            # Copy validation images
            val_save_dir = os.path.join(OUTPUT_DIR, "val", class_id_str)
            os.makedirs(val_save_dir, exist_ok=True)
            for img_path in val_imgs:
                shutil.copy(img_path, val_save_dir)

            id_counter += 1

    # Create query/gallery sets from the validation set
    for cls, class_id in id_map.items():
        val_dir = os.path.join(OUTPUT_DIR, "val", str(class_id))
        imgs = glob.glob(os.path.join(val_dir, "*.*")) # Get all images
        
        if not imgs:
            continue

        random.shuffle(imgs)
        mid = len(imgs) // 2
        if mid == 0 and len(imgs) == 1: # Handle case with only one validation image
             gallery_imgs = imgs
             query_imgs = []
        else:
            query_imgs = imgs[:mid]
            gallery_imgs = imgs[mid:]

        query_dir = os.path.join(OUTPUT_DIR, "query")
        gallery_dir = os.path.join(OUTPUT_DIR, "gallery")
        os.makedirs(query_dir, exist_ok=True)
        os.makedirs(gallery_dir, exist_ok=True)
        
        for img_path in query_imgs:
            shutil.copy(img_path, query_dir)
        for img_path in gallery_imgs:
            shutil.copy(img_path, gallery_dir)
            
    return id_map

# -----------------------------
# Step2: Dataset 與 Sampler
# -----------------------------
class ShipDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        classes = sorted(os.listdir(root))
        self.cls2idx = {cls:idx for idx, cls in enumerate(classes)}
        for cls in classes:
            # 🌟 關鍵修正：尋找所有常見圖片格式
            img_patterns = [os.path.join(root, cls, "*.jpg"), 
                            os.path.join(root, cls, "*.jpeg"), 
                            os.path.join(root, cls, "*.png")]
            imgs = []
            for pattern in img_patterns:
                imgs.extend(glob.glob(pattern))
                
            for img in imgs:
                self.samples.append(img)
                self.labels.append(self.cls2idx[cls])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

class PKSampler(Sampler):
    def __init__(self, labels, P, K):
        self.labels = labels
        self.P = P
        self.K = K
        self.label2indices = {}
        for idx, label in enumerate(labels):
            self.label2indices.setdefault(label, []).append(idx)
        self.labels_set = list(self.label2indices.keys())
    def __iter__(self):
        batch = []
        random.shuffle(self.labels_set)
        for label in self.labels_set:
            idxs = self.label2indices[label]
            if len(idxs) < self.K:
                idxs = np.random.choice(idxs, self.K, replace=True)
            else:
                idxs = random.sample(idxs, self.K)
            batch.extend(idxs)
            if len(batch) >= self.P * self.K:
                yield batch[:self.P*self.K]
                batch = batch[self.P*self.K:]
        # 處理剩餘的 Batch
        if len(batch) >= self.P * self.K:
            yield batch[:self.P*self.K]
    def __len__(self):
        return len(self.labels)//(self.P*self.K)

# -----------------------------
# Step3: Model
# -----------------------------
class ResNet50Embedding(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
    def forward(self, x):
        feat = self.backbone(x)
        feat = F.normalize(feat, p=2, dim=1)  # L2 normalize
        return feat

# -----------------------------
# Step4: Batch-Hard Triplet Loss
# -----------------------------
def batch_hard_triplet_loss(embeddings, labels, margin):
    dist = torch.cdist(embeddings, embeddings, p=2)
    mask_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
    mask_neg = labels.unsqueeze(1) != labels.unsqueeze(0)
    hardest_pos = (dist * mask_pos.float()).max(1)[0]
    dist_with_inf = dist + 1e5*mask_pos.float()  # ignore pos
    hardest_neg = (dist_with_inf * mask_neg.float()).min(1)[0]
    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss.mean()

# -----------------------------
# Step5: Transform
# -----------------------------
transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -----------------------------
# 🌟 關鍵修正：將主要執行邏輯放入 main 函數
# -----------------------------
def main():
    # 執行資料準備 (檔案移動)
    id_map = prepare_data(DATA_ROOT)
    
    # Step6: DataLoader
    train_dataset = ShipDataset(os.path.join(OUTPUT_DIR,"train"), transform_train)
    
    # 修正：檢查 train_dataset 是否有足夠的資料
    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Check DATA_ROOT path or image extensions.")
        return

    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=PKSampler(train_dataset.labels, BATCH_P, BATCH_K), 
        num_workers=4 # 如果仍有 RuntimeError, 請將 num_workers 改為 0
    )

    if len(train_loader) == 0:
        print(f"Error: DataLoader length is 0. Check if there are enough classes/images for BATCH_P={BATCH_P} and BATCH_K={BATCH_K}. Try reducing BATCH_P or BATCH_K.")
        return

    # Step7: Training Loop
    model = ResNet50Embedding(pretrained=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # 修正：加入 try/except 處理可能的 DataLoader 錯誤
        try:
            for imgs, labels in tqdm(train_loader):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                feats = model(imgs)
                loss = batch_hard_triplet_loss(feats, labels, MARGIN)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss/len(train_loader):.4f}")
        except Exception as e:
            print(f"An error occurred during training epoch {epoch+1}: {e}")
            # 如果訓練中斷，先儲存模型權重
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR,f"resnet50_ship_epoch_{epoch+1}.pth"))
            break
            
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR,"resnet50_ship.pth"))

    # Step8: 特徵抽取 + 建立 FAISS index
    # ... (extract_features 函數內容保持不變) ...
    def extract_features(model, folder):
        model.eval()
        features = []
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        for f in tqdm(files):
            img = Image.open(f).convert("RGB")
            img = transform_test(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model(img).cpu().numpy()
            features.append(feat[0])
            np.save(f+".npy", feat[0])
        features = np.array(features).astype("float32")
        return features, files
        
    gallery_features, gallery_files = extract_features(model, os.path.join(OUTPUT_DIR,"gallery"))
    
    if len(gallery_features) == 0:
        print("Error: Gallery dataset is empty. Cannot build FAISS index.")
        return
        
    index = faiss.IndexFlatL2(gallery_features.shape[1])
    index.add(gallery_features)
    faiss.write_index(index, os.path.join(OUTPUT_DIR,"gallery.index"))
    with open(os.path.join(OUTPUT_DIR,"gallery_files.json"),"w") as f:
        json.dump(gallery_files,f)

    # Step9: Query 比對
    # ... (query_image 函數內容保持不變) ...
    def query_image(model, img_path, index, gallery_files, topk=5, threshold=THRESHOLD):
        model.eval()
        img = Image.open(img_path).convert("RGB")
        img = transform_test(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model(img).cpu().numpy().astype("float32")
        D, I = index.search(feat, topk)
        results = []
        for d, i in zip(D[0], I[0]):
            same = d < threshold
            results.append((gallery_files[i], float(d), same))
        return results

    # 範例 query
    index = faiss.read_index(os.path.join(OUTPUT_DIR,"gallery.index"))
    with open(os.path.join(OUTPUT_DIR,"gallery_files.json"),"r") as f:
        gallery_files = json.load(f)

    query_folder = os.path.join(OUTPUT_DIR,"query")
    query_files = os.listdir(query_folder)
    
    if not query_files:
        print("Warning: Query folder is empty. Skipping example query.")
        return

    query_path = os.path.join(query_folder, query_files[0])
    results = query_image(model, query_path, index, gallery_files)
    print("Query Results:")
    for r in results:
        print(r)


if __name__ == '__main__':
    # 確保所有執行代碼都在此處呼叫 main()
    main()
