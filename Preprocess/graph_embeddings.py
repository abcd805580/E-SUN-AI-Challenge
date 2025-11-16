import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import Node2Vec as PyGNode2Vec
try:
    from torch_geometric.utils import coalesce
except ImportError:
    coalesce = None

def build_undirected_edge_index(G, df_all):
    """
    從有向圖 G 建立無向圖的 edge_index 和鄰居列表 (neigh_idx)。
    
    Args:
        G (nx.DiGraph): NetworkX 有向圖。
        df_all (pd.DataFrame): 包含所有 'acct' 的 DataFrame。

    Returns:
        tuple: (edge_index, neigh_idx)
            - edge_index (torch.Tensor): 無向圖的邊索引。
            - neigh_idx (list of lists): 每個節點的鄰居索引列表。
    """
    accts = df_all['acct'].astype(str).tolist()
    id_map = {a:i for i,a in enumerate(accts)}
    neigh_idx = [[] for _ in range(len(accts))]
    src, dst = [], []
    for u, v, _ in G.edges(data=True):
        if u in id_map and v in id_map:
            iu, iv = id_map[u], id_map[v]
            src += [iu, iv]
            dst += [iv, iu]
            neigh_idx[iu].append(iv)
            neigh_idx[iv].append(iu)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    print(f"[EdgeIndex] undirected edges={edge_index.size(1):,}")
    return edge_index, neigh_idx

def train_node2vec_embeddings_pyg_gpu(
    edge_index,
    num_nodes,
    embedding_dim=128,
    walk_length=20,
    context_size=10,
    walks_per_node=50,
    p=1.0, q=0.75,
    batch_size=1024,
    epochs=8
):
    """
    穩定版 Node2Vec 訓練：
    - 不用 sparse（避免 SparseAdam + CUDA 非法存取）
    - 用 Adam
    - num_workers=0（避免 dataloader 在 Colab/CUDA 上的閒置工作執行緒問題）
    - edge_index contiguous + coalesce 保底安全
    - 若 GPU 出錯，會自動 fallback 到 CPU（只影響這一步；GNN 仍用 GPU）

    Args:
        edge_index (torch.Tensor): 邊索引。
        num_nodes (int): 節點總數。
        embedding_dim (int): 嵌入維度。
        walk_length (int): 隨機遊走長度。
        context_size (int): Skip-gram 窗口大小。
        walks_per_node (int): 每個節點的遊走次數。
        p (float): Node2Vec return 參數。
        q (float): Node2Vec in-out 參數。
        batch_size (int): 訓練時的 batch size。
        epochs (int): 訓練 epoch 數。

    Returns:
        np.array: 訓練好的 Node2Vec 嵌入 (N, embedding_dim)，已標準化。
    """
    # ---- 邊索引安全處理：確保 long / contiguous / coalesced
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.to(dtype=torch.long, device='cpu').contiguous()
    # coalesce（無權重版，單純把重複邊合併）
    if coalesce is not None:
        try:
            edge_index = coalesce(edge_index, num_nodes=num_nodes)
        except Exception:
            pass  # 沒有就跳過

    def _train_on(device_str):
        device = torch.device(device_str)
        model = PyGNode2Vec(
            edge_index.to(device),
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=p, q=q,
            sparse=False              # ← 關鍵：不用 sparse
        ).to(device)

        # DataLoader：num_workers=0 最穩
        loader = model.loader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # ← 用 Adam

        print(f"[Node2Vec] Training on {device} (epochs={epochs}, batch={batch_size}, sparse=False)")
        model.train()
        for ep in range(1, epochs + 1):
            total = 0.0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total += float(loss)
            avg_loss = total / max(1, len(loader))
            print(f"  Epoch {ep:03d}: loss={avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            emb = F.normalize(model.embedding.weight.detach(), p=2, dim=1)
        return emb.detach().cpu().numpy()

    # 先嘗試 GPU；失敗就 fallback CPU（只這一步）
    try:
        if torch.cuda.is_available():
            return _train_on('cuda')
        else:
            return _train_on('cpu')
    except RuntimeError as e:
        print(f"[Node2Vec] GPU 失敗，改用 CPU 重新訓練；原因：{e}")
        torch.cuda.empty_cache()
        return _train_on('cpu')

def care_topk_edge_index_using_emb(neigh_idx, emb, topk=15):
    """
    使用 Node2Vec 嵌入 (emb) 來計算鄰居間的餘弦相似度，
    並只保留每個節點 Top-K 相似的邊 (CARE-like)。

    Args:
        neigh_idx (list of lists): 鄰居列表。
        emb (np.array): Node2Vec 嵌入。
        topk (int): 要保留的鄰居數量。

    Returns:
        tuple: (edge_index, edge_weight)
            - edge_index (torch.Tensor): 篩選後的邊索引。
            - edge_weight (torch.Tensor): 篩選後的邊權重 (相似度)。
    """
    src, dst, w = [], [], []
    for i in range(len(neigh_idx)):
        neigh = neigh_idx[i]
        if not neigh:
            continue
        vi = emb[i]
        sims = [(j, float(np.dot(vi, emb[j]))) for j in neigh]
        sims = sorted(sims, key=lambda x:x[1], reverse=True)[:min(topk,len(sims))]
        for j, s in sims:
            if s>0:
                src.append(i); dst.append(j); w.append(s)
    src2 = src + dst
    dst2 = dst + src
    w2 = w + w
    print(f"[CARE] filtered edges={len(src2):,} (topk={topk})")
    return torch.tensor([src2,dst2],dtype=torch.long), torch.tensor(w2,dtype=torch.float)


def get_cached_edges(G, df_all, CACHE_DIR, N2V_PARAMS, TOPK_NEIGHBORS):
    """
    載入或計算(並快取) Node2Vec 嵌入和 CARE 篩選後的邊。
    
    Args:
        G (nx.DiGraph): 原始圖。
        df_all (pd.DataFrame): 節點特徵。
        CACHE_DIR (str): 快取檔案儲存路徑。
        N2V_PARAMS (dict): Node2Vec 訓練參數。
        TOPK_NEIGHBORS (int): CARE-TopK 參數。

    Returns:
        tuple: (n2v_emb, edge_index, edge_weight)
    """
    emb_path = os.path.join(CACHE_DIR, "node2vec_emb.npy")
    care_path = os.path.join(CACHE_DIR, "care_edges.pt")

    if os.path.exists(emb_path) and os.path.exists(care_path):
        print("[Cache] Loading Node2Vec embedding & CARE edges ...")
        n2v_emb = np.load(emb_path)
        cache = torch.load(care_path)
        return n2v_emb, cache['edge_index'], cache['edge_weight']

    # 1) 建立 undirected edge_index + 鄰居列表
    edge_index_und, neigh_idx = build_undirected_edge_index(G, df_all)

    # 2) 訓練 Node2Vec（穩定版；自動 GPU/CPU）
    n2v_emb = train_node2vec_embeddings_pyg_gpu(
        edge_index_und,
        num_nodes=len(df_all),
        embedding_dim=N2V_PARAMS['N2V_DIM'],
        walk_length=N2V_PARAMS['N2V_WALK_LEN'],
        context_size=N2V_PARAMS['N2V_WINDOW'],
        walks_per_node=N2V_PARAMS['N2V_NUM_WALKS'],
        p=N2V_PARAMS['N2V_P'], 
        q=N2V_PARAMS['N2V_Q'],
        batch_size=1024,
        epochs=8
    )

    # 3) CARE-like 依相似度取 top-k 邊
    edge_index, edge_weight = care_topk_edge_index_using_emb(
        neigh_idx, n2v_emb, topk=TOPK_NEIGHBORS
    )

    # 4) 存快取
    np.save(emb_path, n2v_emb)
    torch.save({'edge_index': edge_index, 'edge_weight': edge_weight}, care_path)
    print(f"[Cache] Saved Node2Vec→{emb_path}, CARE→{care_path}")
    return n2v_emb, edge_index, edge_weight
