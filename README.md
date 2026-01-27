- Description

Ce projet propose un dashboard interactif permettant de prévoir la volatilité d’un actif financier à l’aide d’un modèle LSTM (Long Short-Term Memory).
Les données proviennent de Yahoo Finance, Alpha Vantage et Quandl.
Le dashboard permet de : 
sélectionner un actif financier (ex : AAPL, MSFT, TSLA) ;
visualiser la volatilité réelle vs prédite ;
comparer les données de différentes sources ;
entraîner ou charger un modèle LSTM pré-entraîné.



- Lancement du projet (code bash)

git clone https://github.com/milanbft/dashboard-iafinance.git
cd dashboard-iafinance

python -m venv venv
venv\Scripts\activate     #Windows
source venv/bin/activate  #Linux/macOS

pip install -r requirements.txt

export ALPHA_VANTAGE_API_KEY="votre_clef_alpha"
export QUANDL_API_KEY="votre_clef_quandl"

streamlit run app.py
