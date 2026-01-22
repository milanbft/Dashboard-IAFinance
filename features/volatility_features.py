#FEATURE ENGINEERING

import numpy as np

def compute_volatility(df,
                       window=30): #volatilité mensuelle

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)) #calcul des rendements logarithmiques (r(t)=ln(P(t)/P(t-1)))

    df["volatility"] = df["log_return"].rolling(window).std() * np.sqrt(252) #calcul de la volatilité glissante (sur window jours) et annualisation de la volatilité (252 jours de bourse par an)
                                                                             #volatilité annuelle=volatilité journalière * (252**(1/2))
    return df.dropna() #supprime les valeurs manquantes

