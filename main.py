import os, gc, math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# 匯入我們拆分好的模組
from Preprocess.data_preprocess import (
    load_csv,
    build_features_and_graph,
    to_pyg_data,
    make_label_and_split,
)
from Preprocess.graph_embeddings import get_cached_edges
from Model.sage_model import SAGEModel

# ============================================\n
# 0. 全域參數設定 (更新版：使用相對路徑)
# ============================================\n

# 取得此 main.py 檔案所在的資料夾絕對路徑
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 建立相對路徑
BASE_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# 自動建立 output 和 cache 資料夾 (如果它們不存在)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 最終的 CSV 和 模型 儲存路徑
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "result.csv")
MODEL_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "sage_model_weights.pt")

# --- 您的原始參數 (保持不變) ---
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# GNN 設定
HID_DIM = 640
DROPOUT = 0.43
EPOCHS = 10000
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10000

# Node2Vec 參數
N2V_PARAMS = {
    "N2V_DIM": 256,
    "N2V_WALK_LEN": 30,
    "N2V_NUM_WALKS": 75,
    "N2V_WINDOW": 10,
    "N2V_P": 1.0,
    "N2V_Q": 0.75,
}
TOPK_NEIGHBORS = 8


# ============================================\n
# 6. 模型訓練與輸出 (更新版：加入儲存模型)
# ============================================\n
def train_and_eval(data, y_np, tr_idx, va_idx, te_idx):
    """
    執行模型訓練、驗證和最終預測。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    data = data.to(device)
    y = torch.tensor(y_np, dtype=torch.float, device=device)
    model = SAGEModel(data.num_node_features, HID_DIM, DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos = y[tr_idx].sum().item()
    neg = len(tr_idx) - pos
    pos_w = max(neg / (pos + 1e-6), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))

    bestf1, pat, bestt, bestst = -1, 20, 0.5, None

    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        logit = model(data.x, data.edge_index)
        loss = loss_fn(logit[tr_idx], y[tr_idx])
        loss.backward()
        opt.step()

        model.eval()
        p = torch.sigmoid(logit[va_idx]).detach().cpu().numpy()
        yt = y[va_idx].detach().cpu().numpy()
        ts = np.linspace(0.01, 0.99, 99)
        f1s = [f1_score(yt, (p > t).astype(int)) for t in ts]
        curf1, curt = max(f1s), ts[int(np.argmax(f1s))]

        if curf1 > bestf1:
            bestf1, pat, bestt = curf1, 20, curt
            bestst = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            pat -= 1

        if ep % 5 == 0 or pat <= 0:
            print(
                f"Epoch {ep:05d} loss={loss.item():.4f} valF1={curf1:.4f} best={bestf1:.4f}@{bestt:.2f}"
            )

        if pat <= 0:
            print("Early stopping...")
            break

    model.load_state_dict(bestst)
    model.eval()
    with torch.no_grad():
        logit = model(data.x, data.edge_index)
        ptest = torch.sigmoid(logit[te_idx]).cpu().numpy()

    ypred = (ptest > bestt).astype(int)
    print(f"[Result] bestF1={bestf1:.4f} thr={bestt:.3f}")

    # ===== 儲存模型權重 =====
    torch.save(bestst, MODEL_WEIGHTS_PATH)
    print(f"[Save] Model weights saved to {MODEL_WEIGHTS_PATH}")

    return ypred


def save_submission(path, df_test, df_all, te_idx, ypred):
    """
    將預測結果儲存為競賽格式的 CSV 檔案。
    """
    out = pd.DataFrame({"acct": df_all.iloc[te_idx]["acct"], "label": ypred})
    out = (
        df_test[["acct"]]
        .merge(out, on="acct", how="left")
        .fillna(0)
        .astype({"label": int})
    )
    out.to_csv(path, index=False)
    print(f"[Save] Submission CSV saved to {path}")


# ============================================\n
# 7. 主流程
# ============================================\n
def main():
    """
    主執行流程：載入、前處理、訓練、儲存。
    """
    print(f"[Main] Project Root: {PROJECT_ROOT}")
    print(f"[Main] Loading data from: {BASE_DIR}")

    df_txn, df_alert, df_test = load_csv(BASE_DIR)
    df_all, G = build_features_and_graph(df_txn)

    # Node2Vec & CARE Edge (使用快取)
    # 注意：get_cached_edges 函式需要 CACHE_DIR 參數
    n2v_emb, edge_index, edge_weight = get_cached_edges(
        G, df_all, CACHE_DIR, N2V_PARAMS, TOPK_NEIGHBORS
    )

    data = to_pyg_data(df_all, edge_index, edge_weight)
    y, tr, va, te = make_label_and_split(df_all, df_alert, df_test)

    # 執行訓練，它會自動儲存模型到 MODEL_WEIGHTS_PATH
    ypred = train_and_eval(data, y, tr, va, te)

    # 儲存 CSV 到 OUTPUT_PATH
    save_submission(OUTPUT_PATH, df_test, df_all, te, ypred)

    print(f"[Main] Process finished. Outputs are in: {OUTPUT_DIR}")

    # 清理快取
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
