#Quandl (Nasdaq)

import nasdaqdatalink  #données macroéconomiques, taux d'intérêt, volatilité (VIX),...


class QuandlAPIError(Exception):  #gestion erreurs
    pass


def fetch_quandl(dataset, api_key):
    try:
        nasdaqdatalink.ApiConfig.api_key = api_key  #clé API définie globalement et stockée
        df = nasdaqdatalink.get(dataset)  #API Quandl

        if df.empty:
            raise QuandlAPIError("No data returned")  #pas de données

        return df

    except Exception as e:  #message d'erreur original
        raise QuandlAPIError(str(e))


