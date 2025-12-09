# -*- coding: utf-8 -*-
"""
MAPA ZOOM OUT: NUSSELT vs REYNOLDS (Prandtl 0.7 a 100)
Extrapolação da Eq. 3.9 de Fang et al. (2019)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

# Estilo visual limpo
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

def mapa_nusselt_zoom_out():
    # --- 1. PROPRIEDADES FÍSICAS (Base de cálculo) ---
    k_fluido = 0.68
    mi_fluido = 2.79e-4
    rho_fluido = 958.0
    g = 9.81
    L_char = ((mi_fluido / rho_fluido)**2 / g)**(1/3)
    fator_conversao = L_char / k_fluido

    # --- 2. CONFIGURAÇÃO DO GRID "ZOOM OUT" ---
    # Eixo X: Reynolds (Mantemos a faixa laminar do artigo)
    re_vals = np.logspace(np.log10(100), np.log10(2000), 500)
    
    # Eixo Y: Nusselt (ABRIMOS MUITO A ESCALA)
    # Para comportar Pr=100 com expoente -3.2, o Nusselt cai para a ordem de 10^-6
    # Vamos de 10^-7 até 10^1 (5.0)
    nu_vals = np.logspace(-7, 1, 600)
    
    Re, Nu = np.meshgrid(re_vals, nu_vals)
    
    # --- 3. MATEMÁTICA (Fang et al. Eq 3.9) ---
    # h_o = 5.32e4 * Re^(-0.1418) * Pr^(-3.1975)
    
    Ho_grid = Nu / fator_conversao
    A = 5.32 * 10**4
    exp_Re = -0.1418
    exp_Pr = -3.1975
    
    termo_re = A * (Re ** exp_Re)
    
    # Pr = (Ho / Termo_Re)^(1 / -3.1975)
    Pr_grid = (Ho_grid / termo_re) ** (1 / exp_Pr)

    # --- 4. PLOTAGEM ---
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Definindo os níveis de cor para cobrir de 0.7 até 100
    levels = np.logspace(np.log10(0.7), np.log10(100), 200)
    
    # Contourf
    # Spectral_r inverte as cores: Azul agora é "Frio/Viscoso/Pr Alto" e Vermelho é "Quente/Pr Baixo"
    cf = ax.contourf(Re, Nu, Pr_grid, levels=levels, cmap='Spectral', norm=LogNorm())
    
    # Linhas de contorno específicas que você pediu
    pr_lines = [0.7, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    cp = ax.contour(Re, Nu, Pr_grid, levels=pr_lines, colors='black', linewidths=0.8, alpha=0.6)
    
    # Labels nas linhas
    ax.clabel(cp, inline=True, fontsize=10, fmt='Pr=%.1f', colors='black')

    # --- 5. DETALHES VISUAIS ---
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel(r'Número de Reynolds ($Re$)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Número de Nusselt ($Nu^*$)', fontsize=12, fontweight='bold')
    
    ax.set_title("Visão Global: Impacto do Prandtl (0.7 a 100)\nBaseado em Fang et al. (2019)", 
                 fontsize=15, pad=20, fontweight='bold')
    
    # Barra de cores
    cbar = fig.colorbar(cf, ax=ax, ticks=[0.7, 1, 5, 10, 50, 100])
    cbar.set_label(r'Número de Prandtl ($Pr$)', fontsize=12, rotation=270, labelpad=20)
    cbar.ax.set_yticklabels(['0.7 (Gases)', '1 (Vapor)', '5 (Água)', '10', '50', '100 (Óleos)'])
    
    # --- 6. ANOTAÇÕES DE ZONAS ---
    # Zona Eficiente (Topo)
    ax.text(120, 0.5, "ZONA EFICIENTE\n(Gases e Vapor)\nPr ~ 0.7 - 2.0", 
            color='darkred', fontsize=11, fontweight='bold', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Zona Média
    ax.text(120, 0.002, "ZONA LÍQUIDA\n(Água Fria)\nPr ~ 5 - 10", 
            color='darkorange', fontsize=10, fontweight='bold', ha='left')

    # Zona Ineficiente (Fundo)
    ax.text(120, 0.000002, "ZONA VISCOSA\n(Óleos / Glicerina)\nPr > 50\nTroca térmica quase nula", 
            color='navy', fontsize=11, fontweight='bold', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mapa_nusselt_zoom_out()