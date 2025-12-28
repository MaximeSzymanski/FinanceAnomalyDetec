import yfinance as yf
import pandas as pd
import wikipedia
import os
import time
from datetime import datetime

def get_sp500_tickers():
    """
    Récupère la liste officielle S&P 500 via l'API Wikipedia.
    Utilise l'API pour éviter les erreurs de scraping direct.
    """
    page_title = "List of S&P 500 companies"
    
    try:
        print(f"[{datetime.now()}] 🔍 Interrogation de l'API Wikipedia pour '{page_title}'...")
        
        # 1. Chargement de la page via l'API (auto_suggest=False évite les confusions)
        page = wikipedia.page(page_title, auto_suggest=False)
        
        # 2. Récupération du HTML
        html_content = page.html()
        
        # 3. Parsing des tables avec Pandas
        dfs = pd.read_html(html_content)
        
        # La table principale est la première
        df_sp500 = dfs[0]
        
        # 4. Extraction et Nettoyage
        tickers = df_sp500['Symbol'].tolist()
        
        # Yahoo Finance utilise des tirets (-) au lieu des points (.)
        # Ex: Berkshire Hathaway est BRK.B sur Wiki mais BRK-B sur Yahoo
        tickers = [symbol.replace('.', '-') for symbol in tickers]
        
        print(f"[{datetime.now()}] ✅ {len(tickers)} tickers récupérés depuis Wikipédia.")
        return tickers

    except Exception as e:
        print(f"❌ Erreur lors de la récupération Wiki ({e}). Utilisation de la liste de secours.")
        # Fallback si Wikipedia échoue
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'PG']

def fetch_data(tickers, period="2y"):
    """
    Récupère les données OHLCV de manière séquentielle pour éviter le blocage IP.
    """
    print(f"[{datetime.now()}] 🚀 Démarrage du téléchargement pour {len(tickers)} tickers...")
    
    all_data = []
    
    for i, ticker in enumerate(tickers):
        try:
            # Indicateur de progression visuel
            print(f"[{i+1}/{len(tickers)}] Downloading {ticker}...", end=" ")
            
            # Utilisation de Ticker() individuel (plus robuste que .download groupé)
            dat = yf.Ticker(ticker)
            df = dat.history(period=period)
            
            if df.empty:
                print("⚠️ Vide (Ignoré)")
                continue
            
            # On ne garde que la 'Close'
            df = df[['Close']]
            df.columns = [ticker] # Renomme la colonne
            all_data.append(df)
            print("✅ OK")
            
            # Pause pour respecter le rate limit de Yahoo (Anti-ban)
            time.sleep(0.2) 
            
        except Exception as e:
            print(f"❌ Erreur: {e}")

    if not all_data:
        raise ValueError("Aucune donnée n'a pu être récupérée.")

    print(f"[{datetime.now()}] Fusion des données...")
    combined_df = pd.concat(all_data, axis=1)
    
    # Nettoyage final : Forward Fill puis Backward Fill pour les jours fériés/manquants
    if combined_df.isnull().values.any():
        print("Warning: NaNs détectés. Application d'un fill.")
        combined_df = combined_df.ffill().bfill()
        
    return combined_df

def save_data(df, path="data/raw/market_data.csv"):
    """Sauvegarde les données en CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"[{datetime.now()}] 💾 Données sauvegardées dans {path} | Shape: {df.shape}")

if __name__ == "__main__":
    # 1. Récupération de la liste à jour
    sp500_tickers = get_sp500_tickers()
    
    if sp500_tickers:
        # NOTE : Pour tester rapidement, on prend seulement les 50 premiers tickers.
        # En production, enlève le "[:50]" pour tout télécharger (ça prendra ~3-4 minutes).
        tickers_to_download = sp500_tickers[:500]
        
        # 2. Téléchargement et Sauvegarde
        df = fetch_data(tickers_to_download)
        save_data(df)