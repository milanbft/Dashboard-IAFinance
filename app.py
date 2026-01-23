#APP

import streamlit as st
import numpy as np
import pandas as pd
import torch #Deeplearning, backpropagation,...
import torch.nn as nn #définit les modèles
import matplotlib.pyplot as plt
from collectors.alpha_vantage import fetch_alpha_vantage
from collectors.yahoo import fetch_yahoo
from collectors.quandl import fetch_quandl
from features.volatility_features import compute_volatility
from features.config import *

st.set_page_config(page_title="Volatility Predictor", layout="wide")

st.title("📈 Dashboard IA Finance - Prédiction de Volatilité")

symbol = st.text_input("Ticker de l'actif", value="AAPL") #choix de l'actif

#os.makedirs(DATA_DIR, exist_ok=True) #création dossier de stockage

#ACQUISITION DES DONNEES

#Yahoo Finance
try:
    df_yahoo = fetch_yahoo(symbol) #data
    df_yahoo = compute_volatility(df_yahoo) #calculs
    #df_yahoo.to_csv(f"{DATA_DIR}/{symbol}_yahoo.csv") #sauvegarde
except Exception as e:
    st.error(f"Erreur Yahoo Finance : {e}")
    st.stop()

#Alpha Vantage
try:
    df_av = fetch_alpha_vantage(symbol, ALPHA_VANTAGE_API_KEY) #data
    df_av.rename(columns={"4. close": "Close"}, inplace=True) #renomme car Alpha Vantage renvoie "4. close"
    df_av = compute_volatility(df_av) #calculs
    #df_av.to_csv(f"{DATA_DIR}/{symbol}_alpha_vantage.csv") #sauvegarde
except Exception as e:
    st.warning(f"Erreur Alpha Vantage : {e}")
    df_av = df_yahoo.copy()  #fallback

#Quandl
try:
    df_vix = fetch_quandl("CBOE/VIX", QUANDL_API_KEY) #volatilité (à changer)
    #df_vix.to_csv(f"{DATA_DIR}/VIX_quandl.csv") #sauvegarde
except Exception as e:
    st.warning(f"Erreur Quandl : {e}")
    df_vix = pd.DataFrame(index=df_yahoo.index, data={"VIX": np.zeros(len(df_yahoo))})

#RESEAU DE NEURONES LSTM (Long-Short-Term-Memory)

df = df_yahoo.join(df_av, lsuffix="_yahoo", rsuffix="_av", how="inner") #fusion des dataframes yahoo et Alpha Vantage
                                                                        #(ajoute _yahoo et _av aux colonnes communes et garde uniquement les dates présentes dans les deux dataframes)
df = df.join(df_vix, how="inner") #fusionne avec Quandl (garde uniquement les dates communes)
df = df.dropna(inplace=True) #supprime les valeurs manquantes

X = df[["log_return_yahoo", "log_return_av", "VIX"]].values #sélectionne les inputs (rendements Yahoo et Alpha Vantage et volatilité) : array numpy
y = df["volatility_yahoo"].values #valeur attendue : array numpy

sequence_length = 30 #(window)

def create_sequences(X, y, seq_len): #création séquences LSTM
    xs, ys = [], [] #listes séquences d'entrée (features) et des targets (volatilité)
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len]) #slice des features sur la window
        ys.append(y[i+seq_len]) #slice des prédictions après la séquence
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(X, y, sequence_length)

X_tensor = torch.tensor(X_seq, dtype=torch.float32) #conversion Pytorch
y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1) #prépare la régression continue

class VolatilityLSTM(nn.Module): #définition du modèle
    def __init__(self, input_size, #nombre de features
                 hidden_size=64, #à changer
                 num_layers=2, #nombre de couches LSTM empilées
                 dropout=0.2): #évite l'overfitting
        super(VolatilityLSTM, self).__init__() #initialisation (classe fille de nn)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, #shape data
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1) #transforme l'output en 1 seule valeur (prédiction)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x) #(hidden layers, état de cellule et tensor)
        out = out[:, -1, :]  #dernière sortie temporelle
        out = self.fc(out) #convertion en une seule prédiction
        return out

model = VolatilityLSTM(input_size=X_tensor.shape[2])

#Entraînement LSTM

TRAIN_MODEL = st.checkbox("Entraîner le modèle maintenant ?", value=True)

if TRAIN_MODEL:
    criterion = nn.MSELoss() #Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #algorithme d'optimisation de mis à jours des poids (learning rate à changer)
    epochs = st.slider("Nombre d'époques", min_value=10, max_value=200, value=50, step=10)

    for epoch in range(epochs):
        model.train() #modèle en mode entraînement
        optimizer.zero_grad() #reset des gradients
        output = model(X_tensor) #prédiction pour chaque séquence
        loss = criterion(output, y_tensor) #calcul l'erreur entre l'output et le réel
        loss.backward() #calcul automatique des gradients
        optimizer.step() #mis à jour des poids

        if (epoch+1) % 10 == 0: #affichage périodique
            st.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}") #Loss décroît=apprentissage

else:
    st.info("Chargement du modèle pré-entraîné (à implémenter si disponible)")

#Prédiction

model.eval() #modèle en mode évaluation
with torch.no_grad(): #bloque la création du graphe de calcul
    y_pred = model(X_tensor).numpy() #conversion tensor des prédictions en array

#Possibilité de calculer R²

#Affichage des résultats

st.subheader("Volatilité réelle vs prédite")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(y_seq, label="Volatilité réelle")
ax.plot(y_pred, label="Volatilité prédite")
ax.set_xlabel("Jours")
ax.set_ylabel("Volatilité annualisée")
ax.legend()
st.pyplot(fig)

if st.checkbox("Afficher les données brutes"):
    st.write(df)