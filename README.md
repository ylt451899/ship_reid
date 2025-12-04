# Ship Feature Detection and Re-identification

本專案旨在進行船隻特徵檢測與重識別 (ReID)。專案使用 ResNet50 模型從船隻影像中提取特徵，並基於這些特徵執行重識別任務。

## 專案結構

```
.
├── requirements.txt            # 執行專案所需的 Python 套件清單
├── ship_feature_detection.py   # 船隻特徵檢測與重識別的主要執行腳本
├── train.py                    # 訓練船隻重識別模型的腳本(會先將開源資料集轉換為訓練資料集，再進行訓練)
├── README.md                   # 本說明文件
└── ship_feature/
    ├── gallery_files.json      # 包含 Gallery (圖庫) 影像路徑的 JSON 檔案
    ├── gallery.index           # Gallery 特徵的索引檔 (可能用於高效搜尋)
    └── resnet50_ship.pth       # 用於船隻特徵提取的預訓練 ResNet50 模型權重
```

## 安裝說明

1.  複製 (Clone) 儲存庫:
    ```bash
    git clone <repository-url>
    ```
2.  安裝所需套件:
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

### 訓練 (Training)

若要訓練模型，請執行 train.py 腳本：

```bash
python train.py
```

### 特徵檢測與重識別 (Feature Detection and Re-identification)

若要執行船隻特徵檢測與重識別，請執行 ship_feature_detection.py 腳本：

```bash
python ship_feature_detection.py
```

## 模型 (Model)

本專案使用預訓練的 ResNet50 模型 (resnet50_ship.pth) 進行特徵提取。該模型已在大型船隻影像資料集上進行訓練，能夠學習具備強健性與辨別力的特徵。

## 資料集 (Dataset)

本專案使用的資料集位於 Warships(https://github.com/scott0o0/Warships-Reid?tab=readme-ov-file)。
執行train.py後資料集會被分為訓練集、驗證集、Gallery 集和 Query 集，分別位於 ship_feature/train、ship_feature/val、ship_feature/gallery 和 ship_feature/query 目錄下。
