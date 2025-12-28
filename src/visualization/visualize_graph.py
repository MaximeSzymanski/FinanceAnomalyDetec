import torch
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import to_networkx

def visualize_market_graph(graph_idx=-1):
    """
    Visualise le graphe du marché pour une journée donnée.
    Args:
        graph_idx (int): L'index du jour à visualiser (-1 pour le dernier jour disponible).
    """
    # 1. Chargement des données
    try:
        graphs = torch.load("data/processed/graphs.pt",weights_only=False)
        # On a besoin des noms des tickers pour étiqueter les nœuds
        df = pd.read_csv("data/raw/market_data.csv", index_col=0)
        tickers = df.columns.tolist()
    except FileNotFoundError:
        print("Erreur: Fichiers manquants. Lance 'fetcher.py' puis 'graph_builder.py' d'abord.")
        return

    # 2. Sélection du graphe
    data = graphs[graph_idx]
    date_str = data.timestamp
    print(f"Visualisation du graphe pour la date : {date_str}")
    print(f"Nombre de nœuds : {data.num_nodes} | Nombre d'arêtes : {data.num_edges}")

    # 3. Conversion PyTorch Geometric -> NetworkX
    # to_undirected=True car la corrélation est symétrique
    G = to_networkx(data, to_undirected=True)

    # 4. Mapping des index (0, 1, 2...) vers les Tickers (AAPL, MSFT...)
    mapping = {i: name for i, name in enumerate(tickers)}
    G = nx.relabel_nodes(G, mapping)

    # 5. Dessin
    plt.figure(figsize=(14, 10))
    
    # Layout : Algorithme de force (Spring) pour grouper les nœuds connectés
    pos = nx.spring_layout(G, k=0.3, seed=42) 
    
    # Dessin des nœuds
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#6bc6ff", edgecolors="black")
    
    # Dessin des étiquettes (Tickers)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Dessin des arêtes (Liens de similarité)
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="gray")

    plt.title(f"Market Connectivity Graph (k-NN) - {date_str}", fontsize=16)
    plt.axis('off')
    
    # Sauvegarde et affichage
    output_path = f"market_graph_{date_str}.png"
    plt.savefig(output_path)
    print(f"Graphe sauvegardé sous : {output_path}")
    plt.show()

if __name__ == "__main__":
    # Visualise le dernier jour connu
    visualize_market_graph(-1)