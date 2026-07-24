# 📈 Simulateur de Convergence : Black-Scholes vs Monte Carlo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://monte-carlo-convergence-in-option-pricing-4ye4vbfdybdiczskpluh.streamlit.app/)

## 📌 Projet
Cette application interactive permet de visualiser et d'analyser la tarification d'options européennes (Call et Put) en comparant le modèle analytique exact de Black-Scholes-Merton avec des simulations numériques de Monte Carlo. 

L'objectif est d'explorer empiriquement les lois de convergence mathématique, de modéliser les trajectoires de l'actif sous-jacent et d'analyser les sensibilités de l'option face aux variations du marché.

---

## ✨ Fonctionnalités clés
* **Simulation de trajectoires géométriques browniennes :** Visualisation interactive des chemins du sous-jacent avec intervalle de confiance à 95 %.
* **Étude de la convergence :** Démonstration graphique de la loi des grands nombres et vérification de la vitesse de convergence en $1/\sqrt{N}$.
* **Calcul des Grecques (Sensibilités) :** Extraction en temps réel du Delta, Gamma, Vega, Theta et Rho pour Calls et Puts.
* **Extraction de Volatilité Implicite :** Résolution numérique par l'algorithme de Newton-Raphson pour retrouver la volatilité à partir des prix observés sur le marché.

---

## 🛠️ Stack technique
* **Langage :** Python 3
* **Interface & Data Viz :** Streamlit, Matplotlib
* **Calcul Scientifique :** NumPy, Pandas, SciPy (statistiques et géométrie probabiliste)

---

## 🚀 Installation et exécution locale

1. **Cloner le dépôt :**
   git clone [https://github.com/Leandr75e/Monte-Carlo-Convergence-in-Option-Pricing](https://github.com/Leandr75e/Monte-Carlo-Convergence-in-Option-Pricing)
   cd nom-du-depot

2. **Installer les dépendances :**
    pip install -r requirements.txt

3. **Lancer l'application :**
    streamlit run app.py
