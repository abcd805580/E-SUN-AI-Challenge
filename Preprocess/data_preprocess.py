import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
import torch

def load_csv(dir_path):
    """
    從指定目錄載入 acct_transaction.csv, acct_alert.csv, 和 acct_predict.csv。

    Args:
        dir_path (str): 包含 CSV 檔案的資料夾路徑。

    Returns:
        tuple: (df_txn, df_alert, df_test)
    """
    df_txn = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'))
    df_alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    df_test = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'))
    print(f"[Load] txn={len(df_txn):,}  alert={len(df_alert):,}  predict={len(df_test):,}")
    return df_txn, df_alert, df_test

def build_features_and_graph(df_txn):
    """
    基於交易資料(df_txn)構建節點特徵和有向圖(DiGraph)。

    特徵包括：
    - 發送/接收 交易的 sum, mean, max, min, std, count
    - 發送/接收 比例特徵
    - log1p 轉換
    - PageRank, in/out/total degree, in/out ratio

    Args:
        df_txn (pd.DataFrame): 交易資料。

    Returns:
        tuple: (df_all, G)
            - df_all (pd.DataFrame): 包含所有帳戶及其特徵的 DataFrame。
            - G (nx.DiGraph): 根據交易紀錄建立的有向圖。
    """
    send = df_txn.groupby('from_acct')['txn_amt'].agg(['sum','mean','max','min','std','count']).fillna(0)
    send.columns = [f'send_{c}' for c in send.columns]
    recv = df_txn.groupby('to_acct')['txn_amt'].agg(['sum','mean','max','min','std','count']).fillna(0)
    recv.columns = [f'recv_{c}' for c in recv.columns]
    df_feat = send.reset_index().rename(columns={'from_acct':'acct'}).merge(
        recv.reset_index().rename(columns={'to_acct':'acct'}), on='acct', how='outer').fillna(0)
    df_feat['send_recv_sum_ratio'] = (df_feat['send_sum']+1)/(df_feat['recv_sum']+1)
    df_feat['send_recv_cnt_ratio'] = (df_feat['send_count']+1)/(df_feat['recv_count']+1)
    for c in ['send_sum','recv_sum','send_max','recv_max']:
        df_feat[f'log1p_{c}'] = np.log1p(df_feat[c].clip(lower=0))
    print("[Graph] build DiGraph ...")
    G = nx.from_pandas_edgelist(df_txn, source='from_acct', target='to_acct',
                                edge_attr='txn_amt', create_using=nx.DiGraph())
    pr = nx.pagerank(G, alpha=0.85)
    deg_in, deg_out, deg_all = dict(G.in_degree()), dict(G.out_degree()), dict(G.degree())
    graph_df = pd.DataFrame({
        'acct': list(G.nodes()),
        'pagerank': [pr.get(n,0.0) for n in G.nodes()],
        'in_degree': [deg_in.get(n,0) for n in G.nodes()],
        'out_degree':[deg_out.get(n,0) for n in G.nodes()],
        'total_degree':[deg_all.get(n,0) for n in G.nodes()]
    })
    graph_df['in_out_ratio'] = (graph_df['in_degree']+1)/(graph_df['out_degree']+1)
    df_all = df_feat.merge(graph_df, on='acct', how='outer').fillna(0)
    print(f"[Feature] accounts={len(df_all):,}  features={df_all.shape[1]-1}")
    return df_all, G

def to_pyg_data(df_all, edge_index, edge_weight):
    """
    將 DataFrame 特徵和邊轉換為 PyG (PyTorch Geometric) Data 物件。
    特徵會進行 StandardScaler 標準化。

    Args:
        df_all (pd.DataFrame): 包含特徵的 DataFrame。
        edge_index (torch.Tensor): 邊索引。
        edge_weight (torch.Tensor): 邊權重。

    Returns:
        torch_geometric.data.Data: PyG 資料物件。
    """
    feat_cols = [c for c in df_all.columns if c!='acct']
    X = StandardScaler().fit_transform(df_all[feat_cols].values.astype(np.float32))
    data = Data(x=torch.tensor(X,dtype=torch.float),
                edge_index=edge_index,
                edge_weight=edge_weight)
    data.acct = df_all['acct'].tolist()
    data.feat_cols = feat_cols
    return data

def make_label_and_split(df_all, df_alert, df_test):
    """
    根據 alert 和 test 檔案，生成標籤 (y) 並切分訓練、驗證、測試集的索引。

    Args:
        df_all (pd.DataFrame): 所有帳戶的 DataFrame。
        df_alert (pd.DataFrame): 警告帳戶 (label=1) 的 DataFrame。
        df_test (pd.DataFrame): 預測目標帳戶的 DataFrame。

    Returns:
        tuple: (y, tr_idx, va_idx, te_idx)
            - y (np.array): 全部的標籤陣列。
            - tr_idx (np.array): 訓練集索引。
            - va_idx (np.array): 驗證集索引。
            - te_idx (np.array): 測試集索引。
    """
    y = df_all['acct'].isin(df_alert['acct']).astype(int).values
    is_test = df_all['acct'].isin(df_test['acct']).values

    #  修正這裡：若沒有 is_esun 欄位，就自動補成全 1 陣列
    if 'is_esun' in df_all.columns:
        is_esun = df_all['is_esun'].values
    else:
        is_esun = np.ones(len(df_all), dtype=int)

    pool = (~is_test) & (is_esun==1)
    idx_pool = np.where(pool)[0]
    # 注意：這裡的 RANDOM_STATE 需與 main.py 中的全域變數一致
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42) 
    tr,va = next(skf.split(idx_pool,y[idx_pool]))
    return y, idx_pool[tr], idx_pool[va], np.where(is_test)[0]
