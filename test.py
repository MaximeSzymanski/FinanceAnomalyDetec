import yfinance as yf
import pandas as pd
import wikipedia
import os
import time
from datetime import datetime

def get_sp500_tickers():
    """
    Récupère la liste S&P 500 via l'API Wikipedia (plus robuste que le scraping direct).
    """
    page_title = "List of S&P 500 companies"
    
    try:
        print(f"[{datetime.now()}] Interrogation de l'API Wikipedia pour '{page_title}'...")
        
        # 1. On charge l'objet page via l'API
        # Note: auto_suggest=False évite que l'API devine mal si le titre est ambigu
        page = wikipedia.page(page_title, auto_suggest=False)
        
        # 2. On récupère le HTML complet de la page
        html_content = page.html()
        
        # 3. On parse les tables dans le HTML récupéré
        # Pandas va trouver toutes les tables <table> dans le code HTML
        dfs = pd.read_html(html_content)
        
        # La table principale est généralement la première (index 0)
        df_sp500 = dfs[0]
        
        # 4. Extraction et Nettoyage
        tickers = df_sp500['Symbol'].tolist()
        
        # Correction classique : Yahoo utilise des tirets (-), Wiki des points (.)
        # Ex: Berkshire Hathaway (BRK.B -> BRK-B)
        tickers = [symbol.replace('.', '-') for symbol in tickers]
        
        print(f"[{datetime.now()}] ✅ {len(tickers)} tickers récupérés avec succès via l'API.")
        return tickers

    except wikipedia.exceptions.PageError:
        print(f"❌ Erreur: La page '{page_title}' n'existe pas sur Wikipedia.")
        return []
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"❌ Erreur: Titre ambigu. Options: {e.options}")
        return []
    except Exception as e:
        print(f"❌ Erreur critique lors de la récupération Wiki : {e}")
        # Fallback manuel si l'API échoue
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

def fetch_data(tickers, period="2y"):
    """
    Récupère les données OHLCV de manière séquentielle.
    """
    print(f"[{datetime.now()}] Démarrage du téléchargement pour {len(tickers)} tickers...")
    
    all_data = []
    
    # Pour éviter d'y passer la nuit lors des tests,
    # tu peux décommenter la ligne suivante pour limiter à 10 actions :
    # tickers = tickers[:10] 
    
    for ticker in tickers:
        try:
            print(f"Downloading {ticker}...", end=" ")
            dat = yf.Ticker(ticker)
            df = dat.history(period=period)
            
            if df.empty:
                print("❌ Vide (Bloqué ou ticker invalide)")
                continue
            
            # On ne garde que la 'Close'
            df = df[['Close']]
            df.columns = [ticker]
            all_data.append(df)
            print("✅ OK")
            
            # Pause anti-ban Yahoo
            time.sleep(0.3) 
            
        except Exception as e:
            print(f"❌ Erreur: {e}")

    if not all_data:
        raise ValueError("Aucune donnée récupérée.")

    print("Fusion des données...")
    combined_df = pd.concat(all_data, axis=1)
    
    # Nettoyage final
    if combined_df.isnull().values.any():
        print("Warning: NaNs détectés (marchés fermés/jours fériés différents). Forward Fill appliqué.")
        combined_df = combined_df.ffill().bfill()
        
    return combined_df

def save_data(df, path="data/raw/market_data.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"[{datetime.now()}] Données sauvegardées dans {path} | Shape: {df.shape}")

if __name__ == "__main__":
    # Récupération dynamique via l'API Wikipedia
    sp500_tickers = get_sp500_tickers()
    
   