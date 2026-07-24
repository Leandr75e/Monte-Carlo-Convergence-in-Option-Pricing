import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
import streamlit as st 

st.sidebar.header("Paramètres")

st.sidebar.subheader("Graphique des Trajectoires")
n_sim_plot = st.sidebar.slider("Nombre de courbes à afficher", 10, 200, 50)
n_steps_plot = st.sidebar.slider("Nombre de pas de temps (jours)", 20, 252, 100)
N = st.sidebar.number_input("Nombre de simulations", 1000, 1_000_000, 100_000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("Module Volatilité Implicite")
market_price_input = st.sidebar.number_input("Prix observé de l'option sur le marché", min_value=0.01, value=10.0, step=0.5)

st.sidebar.markdown("---")

st.sidebar.header("Paramètres de l'option")

S0 = st.sidebar.slider("Prix actuel du sous-jacent (S0)", 10.0, 200.0, 100.0)
K = st.sidebar.slider("Prix d'exercice (K)", 10.0, 200.0, 100.0)
T = st.sidebar.slider("Maturité (T en années)", 0.1, 5.0, 1.0)
r = st.sidebar.slider("Taux sans risque (r)", 0.0, 0.15, 0.05)
sigma = st.sidebar.slider("Volatilité (sigma)", 0.05, 1.0, 0.20)




st.title("Simulateur de Convergence : Black-Scholes vs Monte Carlo")

def black_scholes(S0, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S0/K) + (r + sigma**2 / 2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    elif option_type == 'put':
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être call ou put")
    
    return price

def monte_carlo_price(S0, K, T, r, sigma, N, option_type='call', return_paths=False):
    Z = np.random.standard_normal(N)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")
    
    discounted_payoffs = np.exp(-r*T) * payoffs
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(N)

    if return_paths:
        return price, std_error, discounted_payoffs
    return price, std_error

def confidence_interval(price, std_error, confidence=0.95):
    z = norm.ppf(0.5 + confidence/2)  
    lower = price - z * std_error
    upper = price + z * std_error
    return lower, upper

bs_call = black_scholes(S0, K, T, r, sigma, 'call')
bs_put = black_scholes(S0, K, T, r, sigma, 'put')

mc_call, se_call = monte_carlo_price(S0, K, T, r, sigma, N, 'call')
mc_put, se_put = monte_carlo_price(S0, K, T, r, sigma, N, 'put')

ci_call = confidence_interval(mc_call, se_call)
ci_put = confidence_interval(mc_put, se_put)

def generate_paths(S0, T, r, sigma, n_sim, n_steps):
    dt= T/n_steps
    Z = np.random.standard_normal((n_steps, n_sim))
    paths = np.zeros((n_steps+1, n_sim))
    paths[0]=S0

    for t in range(1, n_steps+1):
        paths[t]=paths[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*Z[t-1])
    return paths

def bsm_greeks(S0, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    n_prime_d1 = norm.pdf(d1)

    if option_type=='call':
        delta = norm.cdf(d1)
    else:
        delta=norm.cdf(d1)-1

    gamma= n_prime_d1/(S0*sigma*np.sqrt(T))

    vega = (S0*np.sqrt(T)*n_prime_d1)/100

    theta_call = (-(S0*n_prime_d1*sigma)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365
    theta_put = (-(S0*n_prime_d1*sigma)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    theta = theta_call if option_type =='call' else theta_put

    if option_type=='call':
        rho=(K*T*np.exp(-r*T)*norm.cdf(d2))/100
    else:
        rho=(-K*T*np.exp(-r*T)*norm.cdf(-d2))/100
    
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

def implied_volatility(market_price, S0, K, T, r, option_type='call', max_iter=100, tol=1e-5):
    sigma_init = 0.20
    
    for _ in range(max_iter):
        price = black_scholes(S0, K, T, r, sigma_init, option_type)

        d1 = (np.log(S0 / K) + (r + sigma_init**2 / 2) * T) / (sigma_init * np.sqrt(T))
        vega_raw = S0 * np.sqrt(T) * norm.pdf(d1)
        
        if abs(vega_raw) < 1e-10:
            break
            
        diff = price - market_price
        
        if abs(diff) < tol:
            return sigma_init
            
        sigma_init = sigma_init - diff / vega_raw
        
        if sigma_init <= 0:
            sigma_init = 0.001
            
    return np.nan    
        


N_values = np.unique(np.logspace(1, 5, 20).astype(int))
prices = []
errors = []

for n in N_values:
    p, se = monte_carlo_price(S0, K, T, r, sigma, n, 'call')
    prices.append(p)
    errors.append(se)

prices = np.array(prices)
errors = np.array(errors)

st.subheader("Visualisation des trajectoires Monte Carlo")

time_grid = np.linspace(0, T, n_steps_plot + 1)
paths = generate_paths(S0, T, r, sigma, n_sim_plot, n_steps_plot)

fig_paths, ax = plt.subplots(figsize=(10, 5))

ax.plot(time_grid, paths, linewidth=1, alpha=0.6)

ax.axhline(K, color='black', linestyle='--', linewidth=1.5, label=f'Strike K = {K}')

ax.set_title(f"Simulation de {n_sim_plot} trajectoires du sous-jacent")
ax.set_xlabel("Temps (années)")
ax.set_ylabel("Prix de l'actif ($S_t$)")
ax.grid(alpha=0.3)
ax.legend()

st.pyplot(fig_paths)

# Calculs des modules complémentaires
greeks_call = bsm_greeks(S0, K, T, r, sigma, 'call')
greeks_put = bsm_greeks(S0, K, T, r, sigma, 'put')

iv_computed = implied_volatility(market_price_input, S0, K, T, r, 'call')

# Affichage des Grecques
st.subheader("Les Grecques (Sensibilités Black-Scholes)")
tab1, tab2 = st.tabs(["CALL", "PUT"])

with tab1:
    col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
    col_g1.metric("Delta (Δ)", f"{greeks_call['Delta']:.3f}", help="Sensibilité au prix du sous-jacent")
    col_g2.metric("Gamma (Γ)", f"{greeks_call['Gamma']:.3f}", help="Accélération du Delta")
    col_g3.metric("Vega (ν)", f"{greeks_call['Vega']:.3f}", help="Sensibilité à +1% de volatilité")
    col_g4.metric("Theta (θ)", f"{greeks_call['Theta']:.3f}", help="Perte de valeur par jour (Time decay)")
    col_g5.metric("Rho (ρ)", f"{greeks_call['Rho']:.3f}", help="Sensibilité à +1% du taux d'intérêt")

with tab2:
    col_gp1, col_gp2, col_gp3, col_gp4, col_gp5 = st.columns(5)
    col_gp1.metric("Delta (Δ)", f"{greeks_put['Delta']:.3f}")
    col_gp2.metric("Gamma (Γ)", f"{greeks_put['Gamma']:.3f}")
    col_gp3.metric("Vega (ν)", f"{greeks_put['Vega']:.3f}")
    col_gp4.metric("Theta (θ)", f"{greeks_put['Theta']:.3f}")
    col_gp5.metric("Rho (ρ)", f"{greeks_put['Rho']:.3f}")

# Affichage de la volatilité implicite
st.subheader("Volatilité Implicite du Marché")
if np.isnan(iv_computed):
    st.error("Impossible de faire converger la volatilité implicite. Le prix du marché saisi est probablement hors des limites théoriques (arbitrage).")
else:
    st.metric(label=f"Volatilité Implicite extraite (pour un Call à {market_price_input}$)", value=f"{iv_computed * 100:.2f} %")

st.subheader("Comparaison des prix (CALL)")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Prix Black-Scholes (CALL)", value=f"{bs_call:.4f}")
with col2:
    st.metric(label="Prix Monte Carlo (CALL)", value=f"{mc_call:.4f}", delta=f"Écart: {abs(bs_call-mc_call):.4f}")

st.caption(f"Intervalle de confiance à 95% pour le CALL : [{ci_call[0]:.4f}, {ci_call[1]:.4f}]")


st.subheader("Etude de la convergence")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(N_values, prices, color='steelblue', marker='o', markersize=3, label='Price Montecarlo')
ax1.fill_between(N_values, prices - 1.96*errors, prices + 1.96*errors, color='steelblue', alpha=0.2, label='IC 95%')
ax1.axhline(bs_call, color='crimson', linestyle='--', label='Prix Black-Scholes (exact)')
ax1.set_xscale('log')
ax1.set_xlabel('Nombre de Simulations N (échelle log)')
ax1.set_ylabel('Prix du call')
ax1.set_title('Convergence du prix Monte Carlo')
ax1.legend()
ax1.grid(alpha=0.3)

abs_errors = np.abs(prices - bs_call)
mask = abs_errors > 0
ax2 = axes[1]
ax2.plot(N_values[mask], abs_errors[mask], color='darkorange', marker='o', markersize=3, label=r"Erreur MC-Black-Scholes")

ref = abs_errors[mask][0] * np.sqrt(N_values[mask][0] / N_values[mask])
ax2.plot(N_values[mask], ref, color='gray', linestyle='--', label=r"Référence en $1/\\sqrt{N}$")

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Nombre de Simulations N (échelle log)')
ax2.set_ylabel('Erreur Absolue (échelle log)')
ax2.set_title("Vitesse de convergence de l'erreur")
ax2.legend()
ax2.grid(alpha=0.3, which='both')

plt.tight_layout()
st.pyplot(fig)
