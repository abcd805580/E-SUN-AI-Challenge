# 2025 玉山人工智慧公開挑戰賽 - 程式碼說明

這份程式碼是為了重現 2025 玉山人工智慧公開挑戰賽的初賽結果。

## 1. 建模概述

本模型採用圖神經網路 (Graph Neural Network) 來偵測可疑帳戶。整體流程如下：

1.  **特徵工程**：從原始交易資料 (`acct_transaction.csv`) 中，為每個帳戶 (`acct`) 提取統計特徵，包含交易金額的總和、平均、次數等。
2.  **圖譜構建**：
    * 使用交易資料建立一個有向圖 (DiGraph)，並計算 PageRank、In/Out Degree 等圖譜特徵。
    * 訓練 **Node2Vec** 模型來學習節點的拓撲嵌入 (topological embeddings)。
    * 效仿 **CARE-GNN** 的做法，我們使用 Node2Vec 嵌入的餘弦相似度，為每個節點篩選出 Top-K 
        個最相似的鄰居，建立一個新的、更乾淨的圖譜 (`edge_index`)。
3.  **模型訓練**：
    * 使用一個 3 層的 **GraphSAGE** 模型，搭配殘差連接 (Residual Connection) 和 BatchNorm。
    * 模型輸入為 (1) 的統計特徵和 (2) 的圖譜特徵。
    * 使用加權的 BCEWithLogitsLoss (BCE 損失函數) 來處理樣本不平衡問題。
4.  **快取機制**：Node2Vec 訓練和 CARE-TopK 邊篩選是計算最耗時的部分。程式碼會將這些結果快取 (cache) 在 `CACHE_DIR` 目錄下，第二次執行時將直接載入，大幅加速 `main.py` 的執行。

## 2. 環境需求

### Python 版本
* [cite_start]Python 3.11 [cite: 11]

### 安裝步驟
強烈建議使用 `conda` 或 `virtualenv` 建立虛擬環境。

1.  **安裝 PyTorch (cu121)**
    （此步驟對應您 Notebook 中的 PyG 特定安裝指令）
    ```bash
    # (如果環境中有舊版，先卸載)
    # pip uninstall -y torch torchvision torchaudio torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric pyg-lib
    
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric pyg-lib -f [https://data.pyg.org/whl/torch-2.3.0+cu121.html](https://data.pyg.org/whl/torch-2.3.0+cu121.html)
    ```

2.  **安裝其他 Python 套件**
    接著，安裝 `requirements.txt` 中的其他依賴：
    ```bash
    pip install -r requirements.txt
    ```

## 3. 專案結構
此專案結構依照競賽規範建立 

## 4. (模型) 使用方法

1.  **修改路徑**：
    打開 `main.py` 檔案，修改最上方的三個全域參數路徑：
    * `BASE_DIR`：指向包含 `acct_transaction.csv` 等檔案的**資料夾**。
    * `OUTPUT_PATH`：指定您希望儲存 `result.csv` 的**完整路徑**。
    * `CACHE_DIR`：指定一個資料夾路徑，用於儲存 Node2Vec 嵌入和圖譜快取。

2.  **執行程式**：
    在專案根目錄下，執行 `main.py`：
    ```bash
    python main.py
    ```

    **首次執行**：程式會需要較長時間 (10-20 分鐘不等，依硬體而定) 來訓練 Node2Vec 並建立快取。
    **後續執行**：程式會跳過 Node2Vec 訓練，直接載入快取，僅執行 GraphSAGE 訓練 (約 2-5 分鐘)。

## 5. 實驗結果

### 超參數設定 [cite: 13]
* **GraphSAGE**
    * `HID_DIM`: 640
    * `DROPOUT`: 0.43
    * `EPOCHS`: 10000 (有 Early Stopping)
    * `LR`: 1e-3
    * `WEIGHT_DECAY`: 1e-5
* **Node2Vec**
    * `N2V_DIM`: 256
    * `N2V_WALK_LEN`: 30
    * `N2V_NUM_WALKS`: 75
    * `N2V_P`: 1.0
    * `N2V_Q`: 0.75
* **CARE-TopK**
    * `TOPK_NEIGHBORS`: 8

### 復現結果
在上述設定下，於驗證集 (Validation Set) 上可達到的最佳 F1-Score：
* **Best F1 (Val)**: 0.5291
* **Threshold**: 0.990
