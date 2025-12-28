import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

# Configuration du Split
TRAIN_RATIO = 0.8  # 80% des actions pour l'entraînement, 20% pour le test

def load_data(path="data/raw/market_data.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier {path} introuvable. Lance fetcher.py.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def build_graphs_for_tickers(df, subset_name, window_size=30, k=5):
    """
    Construit une série de graphes pour un sous-ensemble d'actions donné.
    """
    graphs = []
    dates = df.index
    
    # Calcul des rendements
    returns = df.pct_change().fillna(0)
    
    print(f"   🔨 Construction pour '{subset_name}' ({df.shape[1]} tickers) sur {len(dates)} jours...")

    for t in range(window_size, len(dates)):
        # 1. Node Features (Fenêtre glissante)
        current_window = returns.iloc[t-window_size:t]
        
        # Shape: [Num_Nodes_Subset, Window_Size]
        x = torch.tensor(current_window.values.T, dtype=torch.float)
        
        # 2. Edges (Calculés uniquement entre les actions du sous-ensemble)
        correlation_matrix = current_window.corr().fillna(0).values
        
        # Ajustement de k si le subset est trop petit
        current_k = min(k, len(df.columns) - 1)
        if current_k < 1: 
            # Cas rare ou il n'y a qu'une seule action dans le set
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            adj_matrix = kneighbors_graph(correlation_matrix, current_k, mode='connectivity', include_self=False)
            coo = adj_matrix.tocoo()
            row = torch.from_numpy(coo.row.astype(np.int64))
            col = torch.from_numpy(coo.col.astype(np.int64))
            edge_index = torch.stack([row, col], dim=0)
        
        timestamp = str(dates[t].date())
        
        data = Data(x=x, edge_index=edge_index, timestamp=timestamp)
        graphs.append(data)
        
    return graphs

if __name__ == "__main__":
    # 1. Chargement global
    df_full = load_data()
    all_tickers = df_full.columns.tolist()
    
    # 2. Mélange et Split des Tickers (Asset-based Split)
    np.random.seed(42) # Pour la reproductibilité
    np.random.shuffle(all_tickers)
    
    split_idx = int(len(all_tickers) * TRAIN_RATIO)
    train_tickers = all_tickers[:split_idx]
    test_tickers = all_tickers[split_idx:]
    
    print(f"Initialisation du Split par Actifs :")
    print(f" - Train Tickers ({len(train_tickers)}): {train_tickers[:5]}...")
    print(f" - Test Tickers  ({len(test_tickers)}): {test_tickers[:5]}...")
    
    # Création des DataFrames partiels
    df_train = df_full[train_tickers]
    df_test = df_full[test_tickers]
    
    # 3. Construction des graphes
    print("\nConstruction des graphes TRAIN...")
    train_graphs = build_graphs_for_tickers(df_train, "Train Set")
    
    print("\nConstruction des graphes TEST...")
    test_graphs = build_graphs_for_tickers(df_test, "Test Set")
    
    # 4. Sauvegarde séparée
    os.makedirs("data/processed", exist_ok=True)
    
    torch.save(train_graphs, "data/processed/train_graphs.pt")
    torch.save(test_graphs, "data/processed/test_graphs.pt")
    
    print(f"\n✅ Sauvegarde terminée.")
    print(f"   -> data/processed/train_graphs.pt ({len(train_graphs)} jours)")
    print(f"   -> data/processed/test_graphs.pt ({len(test_graphs)} jours)")