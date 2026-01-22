#Yahoo Finance

import yfinance as yf #prix

class YahooFinanceError(Exception): #gestion erreurs
    pass

def fetch_yahoo( #aucune clé API requise
        symbol, #ticker
        start="2000-01-01"): #début de l'historique
    try:
        ticker = yf.Ticker(symbol) #centralise les données de l'actif (dividendes, splits,...)
        df = ticker.history(start=start) #récupère l'historique (OHLC, dividendes, splits, volume)

        if df.empty:
            raise YahooFinanceError("No data returned") #pas de données

        return df

    except Exception as e: #récupère le message d'erreur original
        raise YahooFinanceError(str(e))

