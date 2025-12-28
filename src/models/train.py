import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import numpy as np
import os
from gnn_model import GNNAutoEncoder

MLFLOW_EXPERIMENT_NAME = "Market_Anomaly_Asset_Split"

def load_datasets():
    """Charge les datasets pré-splités par tickers."""
    if not os.path.exists("data/processed/train_graphs.pt"):
        raise FileNotFoundError("Données manquantes. Lance graph_builder.py d'abord.")
        
    print("Chargement des datasets...")
    train_graphs = torch.load("data/processed/train_graphs.pt", weights_only=False)
    test_graphs = torch.load("data/processed/test_graphs.pt", weights_only=False)
    return train_graphs, test_graphs

def evaluate(model, graphs, criterion, device):
    """Fonction helper pour calculer l'erreur moyenne sur un set de graphes."""
    model.eval()
    total_error = 0
    with torch.no_grad():
        for data in graphs:
            data = data.to(device)
            rec, _ = model(data.x, data.edge_index)
            loss = criterion(rec, data.x)
            total_error += loss.item()
    return total_error / len(graphs)

def train(epochs=100, hidden_dim=64, latent_dim=16, lr=0.001, eval_every=1):
    # 1. Setup Data
    train_graphs, test_graphs = load_datasets()
    
    # Feature dim
    num_features = train_graphs[0].x.shape[1]
    
    # 2. Setup MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        # Log Hyperparams
        mlflow.log_param("split_type", "asset_based")
        mlflow.log_param("train_assets_count", train_graphs[0].num_nodes)
        mlflow.log_param("test_assets_count", test_graphs[0].num_nodes)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("lr", lr)

        # Init Model
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        model = GNNAutoEncoder(num_features, hidden_dim, latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        print(f"Démarrage de l'entraînement sur {device}...")
        print(f"Train Set: {len(train_graphs)} jours | Test Set: {len(test_graphs)} jours")
        
        # 3. Training Loop
        for epoch in range(1, epochs + 1):
            model.train()
            total_train_loss = 0
            
            for data in train_graphs:
                data = data.to(device)
                optimizer.zero_grad()
                
                reconstructed, _ = model(data.x, data.edge_index)
                loss = criterion(reconstructed, data.x)
                
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_graphs)
            
            # --- Evaluation Périodique ---
            if epoch % eval_every == 0:
                # Calcul de l'erreur sur le Test Set (Généralisation)
                avg_test_loss = evaluate(model, test_graphs, criterion, device)
                
                print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Test Loss (Unseen Assets): {avg_test_loss:.6f}")
                
                # Logging MLflow (Tu auras deux courbes dans l'UI)
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("test_loss", avg_test_loss, step=epoch)

        # 4. Final Threshold Calculation
        print("\nCalcul du seuil final d'anomalie...")
        model.eval()
        train_errors = []
        with torch.no_grad():
            for data in train_graphs:
                data = data.to(device)
                rec, _ = model(data.x, data.edge_index)
                train_errors.append(criterion(rec, data.x).item())
        
        threshold = np.percentile(train_errors, 95)
        print(f"Seuil d'anomalie défini à : {threshold:.6f}")
        mlflow.log_param("anomaly_threshold", threshold)

        # Sauvegarde
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/gnn_ae_asset_split.pth")
        
if __name__ == "__main__":
    train()