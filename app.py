#APP

import os
import streamlit as st
import numpy as np
import pandas as pd
import torch #Deeplearning, backpropagation,...
import torch.nn as nn #définit les modèles
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #normalisation,...
from collectors.alpha_vantage import fetch_alpha_vantage
from collectors.yahoo import fetch_yahoo
from collectors.quandl import fetch_quandl
from features.volatility_features import compute_volatility
from features.config import *

st.set_page_config(page_title="Volatility Predictor", layout="wide")

st.title("📈 Dashboard IA Finance - Prédiction de Volatilité")

symbol = st.text_input("Ticker de l'actif", value="AAPL") #choix de l'actif

os.makedirs(DATA_DIR, exist_ok=True) #création dossier de stockage

#ACQUISITION DES DONNEES

#Yahoo Finance
try:
    df_yahoo = fetch_yahoo(symbol) #data
    df_yahoo = compute_volatility(df_yahoo) #calculs
    df_yahoo.to_csv(f"{DATA_DIR}/{symbol}_yahoo.csv") #sauvegarde
except Exception as e:
    st.error(f"Erreur Yahoo Finance : {e}")
    st.stop()

#Alpha Vantage
try:
    df_av = fetch_alpha_vantage(symbol, ALPHA_VANTAGE_API_KEY) #data
    df_av.rename(columns={"4. close": "Close"}, inplace=True) #renomme car Alpha Vantage renvoie "4. close"
    df_av = compute_volatility(df_av) #calculs
    df_av.to_csv(f"{DATA_DIR}/{symbol}_alpha_vantage.csv") #sauvegarde
except Exception as e:
    st.warning(f"Erreur Alpha Vantage : {e}")
    df_av = df_yahoo.copy()  #fallback

#Quandl
try:
    df_vix = fetch_quandl("CBOE/VIX", QUANDL_API_KEY) #volatilité (à changer)
    df_vix.to_csv(f"{DATA_DIR}/VIX_quandl.csv") #sauvegarde
except Exception as e:
    st.warning(f"Erreur Quandl : {e}")
    df_vix = pd.DataFrame(index=df_yahoo.index, data={"VIX": np.zeros(len(df_yahoo))})

#RESEAU DE NEURONES LSTM (Long-Short-Term-Memory)

df = df_yahoo.join(df_av, lsuffix="_yahoo", rsuffix="_av", how="inner") #fusion des dataframes yahoo et Alpha Vantage
                                                                        #(ajoute _yahoo et _av aux colonnes communes et garde uniquement les dates présentes dans les deux dataframes)
df = df.join(df_vix, how="inner") #fusionne avec Quandl (garde uniquement les dates communes)
df = df.dropna() #supprime les valeurs manquantes

X = df[["log_return_yahoo", "log_return_av", "VIX"]].values #sélectionne les inputs (rendements Yahoo et Alpha Vantage et volatilité) : array numpy
y = np.log(df["volatility_yahoo"].values + 1e-6) #valeur attendue : array numpy et log de volatility (compresse les pics pour qu'HuberLoss agisse sur des écarts relatifs)

scaler_X = StandardScaler() #moyenne nulle et écart-type égal à 1
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X) #normalisation pour delta=1 pour HuberLoss
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel() 

sequence_length = 30 #window

def create_sequences(X, y, seq_len): #création séquences LSTM
    xs, ys = [], [] #listes séquences d'entrée (features) et des targets (volatilité)
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len]) #slice des features sur la window
        ys.append(y[i+seq_len]) #slice des prédictions après la séquence
    return np.array(xs), np.array(ys)

#Pipeline : volatilité réelle -> log(vol) -> scaling -> prédiction LSTM -> inverse scaling -> exp 

X_seq, y_seq = create_sequences(X, y, sequence_length)

train_size = int(0.8 * len(X_seq))

X_train = X_seq[:train_size] #ajuste le poids du réseau
y_train = y_seq[:train_size]

X_test = X_seq[train_size:] #simule une prédiction
y_test = y_seq[train_size:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

X_test = torch.tensor(X_test, dtype=torch.float32) #conversion Pytorch
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1) #prépare la régression continue

class VolatilityLSTM(nn.Module): #définition du modèle
    def __init__(self, input_size, #nombre de features
                 hidden_size=128, #à changer
                 num_layers=2, #nombre de couches LSTM empilées
                 dropout=0.1, #évite l'overfitting
                 batch_first=True): 
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

model = VolatilityLSTM(input_size=X_test.shape[2])

#Entraînement LSTM

TRAIN_MODEL = st.checkbox("Entraîner le modèle maintenant ?", value=True)

if TRAIN_MODEL:
    #criterion = nn.MSELoss() #Mean Squared Error : 1/N*somme des erreurs au carré
    criterion = nn.HuberLoss(delta=1.0) #combine MSE (petites erreurs) et MAE (grosses erreurs (Mean Absolute Error : 1/N*somme des valeurs absolues des erreurs))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #algorithme d'optimisation de mis à jours des poids (learning rate à changer)
    epochs = st.slider("Nombre d'époques", min_value=20, max_value=300, value=100, step=20)

    for epoch in range(epochs):
        model.train() #modèle en mode entraînement
        optimizer.zero_grad() #reset des gradients
        output = model(X_train) #prédiction pour chaque séquence
        loss = criterion(output, y_train) #calcul l'erreur entre l'output et le réel
        loss.backward() #calcul automatique des gradients
        optimizer.step() #mis à jour des poids

        if (epoch+1) % 10 == 0: #affichage périodique
            st.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.5f}") #Loss décroît=apprentissage

else:
    st.info("Chargement du modèle pré-entraîné (à implémenter si disponible)")

#Prédiction

model.eval() #modèle en mode évaluation
with torch.no_grad(): #bloque la création du graphe de calcul
    y_pred_test = model(X_test).numpy() #conversion tensor des prédictions en array
    y_test_np = y_test.numpy()

y_pred_test = scaler_y.inverse_transform(y_pred_test) #inversion (renvoie à l'unité d'origine pour interpréter)
y_test_np = scaler_y.inverse_transform(y_test_np)

y_pred_test = np.exp(y_pred_test) #retour à la volatilité réelle
y_test_np = np.exp(y_test_np)

#Possibilité de calculer R²

#Affichage des résultats

st.subheader("Volatilité réelle vs prédite")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(y_test_np, label="Volatilité réelle", linewidth=2)
ax.plot(y_pred_test, label="Volatilité prédite", linewidth=2)
ax.set_xlabel("Temps")
ax.set_ylabel("Volatilité annualisée")
ax.legend()
ax.grid(True)

st.pyplot(fig)

if st.checkbox("Afficher les données brutes"):
    st.dataframe(df)
