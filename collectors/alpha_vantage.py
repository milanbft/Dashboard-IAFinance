#Alpha Vantage

import requests
import pandas as pd
import time

class AlphaVantageAPIError(Exception): #gestion des erreurs API
    pass

def fetch_alpha_vantage(symbol, #ticker
    api_key, #clé API
    function="TIME_SERIES_DAILY_ADJUSTED", #type de données demandées
    max_retries=3, #nombre d'essais maximum
    sleep_on_limit=60 #temps maximum avant erreur
):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,  #type de données demandées (OHLC+dividendes+splits)
        "symbol": symbol,  #actif ciblé
        "apikey": api_key,  #authentification
        "outputsize": "full"  #historique complet
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10) #appel de l'API : renvoie contenu JSON
            if response.status_code != 200: #200=requête réussie en http (on vérifie que la requête est bien passée)
                raise AlphaVantageAPIError(f"HTTP {response.status_code}") #Erreur HTTP

            data = response.json() #lecture du texte JSON

            if "Note" in data:
                if attempt < max_retries - 1:
                    time.sleep(sleep_on_limit) #Alpha Vantage renvoie un message JSON et non une erreur HTTP
                    continue
                raise AlphaVantageAPIError("API rate limit exceeded") #limite de requêtes atteinte

            if "Error Message" in data:
                raise AlphaVantageAPIError(data["Error Message"]) #erreur symbole ou fonction (mauvais ticker)

            if "Time Series (Daily)" not in data:
                raise AlphaVantageAPIError("Missing time series data") #données absentes

            df = pd.DataFrame(data["Time Series (Daily)"]).T #conversion en dataframe : on isole les séries temporelles (transposée pour que les dates donc les clés JSON soient l'index)
            df = df.astype(float) #conversion enn numérique
            df.index = pd.to_datetime(df.index) #conversion de l'index en dates
            df.sort_index(inplace=True) #tri chronologique (par défaut les dates d'Alpha Vantage sont décroissantes)

            return df

        except requests.exceptions.Timeout:
            raise AlphaVantageAPIError("Request timeout") #si aucune réponse en 10s

        except requests.exceptions.RequestException as e: #récupère le message d'erreur original (classe mère de toutes les erreurs requests)
            raise AlphaVantageAPIError(str(e))

    raise AlphaVantageAPIError("Max retries exceeded") #erreur non dépendante de requests

