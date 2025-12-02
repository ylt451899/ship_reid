import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# -------------------------------
# 配置區（請在此改成你要的路徑、閾值）
# -------------------------------
# 此處把要比對的兩張圖的絕對路徑或相對路徑直接寫死在程式裡
IMG_PATH_1 = "ship_feature/gallery/ZAPADNYY(19).jpg"   # <-- 改成你的第一張圖路徑（寫死）
IMG_PATH_2 = "ship_feature/query/ZAPADNYY(18).jpg"  # <-- 改成你的第二張圖路徑（寫死）

# 如果你有自己訓練 / 取得的模型權重（.pth），把路徑填在下方，否則留 None 使用 torchvision 的 ImageNet weights
MODEL_PATH = None  # e.g. "resnet50_ship.pth" or None

# 判斷閾值（Euclidean distance）：distance < THRESHOLD => 判為「同一艘船」
# 初始建議值：ImageNet ResNet50 作特徵提取時可從 0.8~1.2 試起；若使用 ReID 專用 model 建議用較小值（0.5~0.9）
THRESHOLD = 0.5

# 輸出細節
PRINT_TOPK = 5  # 若想同時列出 gallery top-k（非本程式重點），保留此變數作為擴充

# Device configuration（有 GPU 就用 GPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 圖像預處理（與 ResNet-50 配套）
# -------------------------------
# 這裡使用 256x256 或 224x224 均可（ResNet 對輸入大小彈性）
IMAGE_SIZE = (256, 256)

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),                  # Resize to fixed size（保持比例/失真取決於你的需求）
    transforms.ToTensor(),                          # HWC -> CHW 並轉為 [0,1]
    transforms.Normalize([0.485,0.456,0.406],      # ImageNet mean
                         [0.229,0.224,0.225])      # ImageNet std
])

# -------------------------------
# 模型定義（只保留 backbone, 並輸出 L2-normalized vector）
# -------------------------------
class ResNet50Backbone(nn.Module):
    """
    ResNet-50 backbone wrapper:
      - 使用 torchvision.models.resnet50
      - 移除最後的 fc（classification head），直接輸出 GAP 後的 2048-d 向量
      - 在 forward 中做 L2 normalization（輸出單位向量）
    """
    def __init__(self, use_pretrained_imagenet=True):
        super().__init__()
        # torchvision: 如果 use_pretrained_imagenet=True，會自動下載 ImageNet weights（若可用）
        # 若你之後要改用 weights param style（torchvision >=0.13），可以改用:
        # from torchvision.models import ResNet50_Weights
        # models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = models.resnet50(pretrained=use_pretrained_imagenet)
        # Remove / replace the classification head with identity so forward returns features
        # In torchvision ResNet, .fc is the final linear after GAP
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        # x: B x C x H x W
        feat = self.backbone(x)                   # -> B x 2048
        feat = nn.functional.normalize(feat, p=2, dim=1)  # L2-normalize each row
        return feat

# -------------------------------
# 輔助函式：載入圖、抽特徵
# -------------------------------
def load_image_to_tensor(path):
    """
    1) 用 PIL 讀圖
    2) 轉成 tensor 並做 normalize
    3) 回傳 shape = (1, C, H, W) 的 Tensor（ready for model）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0)   # add batch dim
    return t.to(DEVICE)

@torch.no_grad()
def extract_embedding(model, img_path):
    """
    給 model 與圖片路徑，回傳 1D numpy 向量（已從 torch tensor 轉成 numpy）
    - model: 已在 DEVICE 上
    - img_path: 圖片檔案路徑
    """
    model.eval()
    x = load_image_to_tensor(img_path)      # 1 x C x H x W on DEVICE
    with torch.no_grad():
        emb = model(x)                      # 1 x 2048 (L2 normalized)
    emb = emb.cpu().numpy().reshape(-1)     # 1D numpy array
    return emb

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """計算兩個向量的 Euclidean distance（L2）"""
    return float(np.linalg.norm(a - b))

# -------------------------------
# 主流程（固定圖路徑、不需 CLI）
# -------------------------------
def main():
    # 1) 初始化模型（若有 MODEL_PATH，先建立相容的 model instance，再載入）
    #    我們預設使用 ImageNet pretrained backbone（若要用自己的 checkpoint，請將 MODEL_PATH 指向你的 .pth 並取消下面註解）
    print(f"[INFO] Using device: {DEVICE}")
    model = ResNet50Backbone(use_pretrained_imagenet=True).to(DEVICE)

    # 若你有自己的 checkpoint（例如你訓練的 resnet50_ship.pth），把 MODEL_PATH 改成該路徑（上方），
    # 並取消下一段註解來載入權重：
    if MODEL_PATH:
        # 範例： 若你的 .pth 是 model.state_dict() 的輸出，直接用 load_state_dict 即可
        # 若 .pth 是一個 dict (e.g., {'model': state_dict, ...}), 你可能需要取出 state dict: ckpt['model']
        print(f"[INFO] Loading model weights from: {MODEL_PATH}")
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        # 自動處理常見的 checkpoint 格式
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            state = ckpt['model']
        elif isinstance(ckpt, dict):
            # 如果是直接的 state_dict，使用它
            state = ckpt
        else:
            state = ckpt
        # 嘗試直接載入；如果 keys mismatch，你可能需要先檢查 ckpt.keys()
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            # 若 key 不完全吻合，嘗試 non-strict load（寬容載入）
            print("[WARN] load_state_dict failed in strict mode; trying non-strict load:", e)
            model.load_state_dict(state, strict=False)
        print("[INFO] Model weights loaded (non-strict if needed).")

    # 2) 把兩張圖寫死的路徑印出來，並抽特徵
    print(f"[INFO] Image 1 path: {IMG_PATH_1}")
    print(f"[INFO] Image 2 path: {IMG_PATH_2}")

    emb1 = extract_embedding(model, IMG_PATH_1)   # 1D numpy, L2-normalized
    emb2 = extract_embedding(model, IMG_PATH_2)

    # 3) 計算距離（Euclidean）
    dist = euclidean_distance(emb1, emb2)
    print(f"[RESULT] Euclidean distance = {dist:.6f}")

    # 4) 閾值比較（決定是否為同一艘船）
    #    注意：閾值需要在 validation dataset 上調過（此處提供建議值與說明）
    if dist < THRESHOLD:
        print(f"[DECISION] distance < {THRESHOLD} -> 判定為 SAME (同一艘船)")
    else:
        print(f"[DECISION] distance >= {THRESHOLD} -> 判定為 DIFFERENT (不同船)")

    # 5) 額外輸出（可視化 / debug 用）
    #    顯示 embedding 的前 8 個元素作檢查（方便 debug）
    print("[DEBUG] emb1[:8] =", np.round(emb1[:8], 6))
    print("[DEBUG] emb2[:8] =", np.round(emb2[:8], 6))

    # 如果你想儲存這兩個 embedding 作後續比對，可用 np.save：
    # np.save("emb_img1.npy", emb1)
    # np.save("emb_img2.npy", emb2)

if __name__ == "__main__":
    main()
