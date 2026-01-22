#ACQUISITION DES DONNEES

import os #créer dossiers, gère chemins et rend le script portable
from collectors.alpha_vantage import fetch_alpha_vantage
from collectors.yahoo import fetch_yahoo
from collectors.quandl import fetch_quandl
from features.volatility_features import compute_volatility
from config import *

symbol = "AAPL" #choix de l'actif (à changer)

os.makedirs(DATA_DIR, exist_ok=True) #création dossier de stockage

#Yahoo Finance
df_yahoo = fetch_yahoo(symbol) #data
df_yahoo = compute_volatility(df_yahoo) #calculs
df_yahoo.to_csv(f"{DATA_DIR}/{symbol}_yahoo.csv") #sauvegarde

#Alpha Vantage
df_av = fetch_alpha_vantage(symbol, ALPHA_VANTAGE_API_KEY) #data
df_av.rename(columns={"4. close": "Close"}, inplace=True) #renomme car Alpha Vantage renvoie "4. close"
df_av = compute_volatility(df_av) #calculs
df_av.to_csv(f"{DATA_DIR}/{symbol}_alpha_vantage.csv") #sauvegarde

#Quandl
df_vix = fetch_quandl("CBOE/VIX", QUANDL_API_KEY) #volatilité (à changer)
df_vix.to_csv(f"{DATA_DIR}/VIX_quandl.csv") #sauvegarde


