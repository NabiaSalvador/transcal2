# -*- coding: utf-8 -*-
"""
PARTE FINAL: DIAGRAMA DE ESPESSURA REAL (FANG ET AL.) VS NUSSELT
Referência: Fang, J., Li, K., & Diao, M. (2019). Eq. 2.5

Descrição:
Relaciona a Espessura Física do Filme (mm) com a Transferência de Calor (Nusselt).
A espessura é calculada rigorosamente usando as propriedades do fluido e a Eq. 2.5 do artigo.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')

def calcular_espessura_fang_real(Re):
    """
    Calcula a espessura REAL (mm) usando a Eq. 2.5 de Fang et al. (2019).
    Considerando propriedades de condensado de água saturada (base do artigo).
    
    delta = [ (3 * mu^2 * Re) / (4 * rho^2 * g) ]^(1/3)
    """
    # Propriedades da Água Saturada (Condensado típico a 100°C)
    mu = 2.79e-4  # Pa.s
    rho = 958.0   # kg/m3
    g = 9.81      # m/s2
    
    # Equação 2.5 rearranjada para função de Reynolds
    termo = (3 * (mu**2) * Re) / (4 * (rho**2) * g)
    delta_m = termo**(1/3)
    
    return delta_m * 1000.0 # Retorna em mm

def calcular_nusselt_global(Re, Pr):
    """
    Calcula o Nusselt (Híbrido Laminar/Turbulento)
    """
    Nu_lam = 1.10 * (Re ** (-1/3))
    Nu_turb = 0.0038 * (Re ** 0.4) * (Pr ** 0.65)
    return (Nu_lam**2 + Nu_turb**2)**0.5

def plotar_espessura_fang_vs_nu():
    # 1. Dados Base (Reynolds varia para gerar a espessura)
    Re_vals = np.logspace(np.log10(50), np.log10(30000), 500)
    
    # 2. Calcular Eixo Y: Espessura REAL (mm) baseada no artigo
    Delta_mm = calcular_espessura_fang_real(Re_vals)
    
    # 3. Definição de Prandtl para as linhas
    Pr_lista = [1, 2, 5, 7, 10]
    cores = plt.cm.viridis(np.linspace(0, 0.9, len(Pr_lista)))
    
    # 4. Plotagem
    fig, ax = plt.subplots(figsize=(8, 7))
    
    for i, Pr in enumerate(Pr_lista):
        # Calcular Eixo X: Nusselt
        Nu_vals = calcular_nusselt_global(Re_vals, Pr)
        
        # PLOT: Nusselt (X) vs Espessura Real (Y)
        ax.plot(Nu_vals, Delta_mm, 
                color=cores[i], 
                linewidth=2.5, 
                label=f'Pr = {Pr}')
        
        # Labels nas linhas para ficar elegante
        # Pegamos um ponto onde o filme é mais grosso
        idx = -10 
        ax.text(Nu_vals[idx], Delta_mm[idx], f' Pr={Pr}', 
                color=cores[i], fontsize=9, fontweight='bold', va='center', ha='left')

    # 5. Linha de Referência Laminar (Teórica)
    # Apenas para mostrar o comportamento sem turbulência
    Re_ref = np.linspace(50, 2000, 100)
    Delta_ref = calcular_espessura_fang_real(Re_ref)
    Nu_ref = 1.10 * (Re_ref ** (-1/3))
    
    ax.plot(Nu_ref, Delta_ref, color='black', linestyle='--', alpha=0.5, label='Teoria Laminar Pura')
    ax.text(Nu_ref[10], Delta_ref[10], "Regime Laminar\n(Nusselt Puro)", 
            fontsize=8, color='black', alpha=0.7, ha='right')

    # 6. Formatação
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Eixos
    ax.set_ylabel(r'Espessura do Filme Calculada - Fang Eq. 2.5 ($\delta$ [mm])', fontsize=11, fontweight='bold')
    ax.set_xlabel(r'Número de Nusselt Global ($Nu^*$)', fontsize=12, fontweight='bold')
    
    ax.set_title("Relação Espessura vs. Nusselt\n(Baseado na Física de Fang et al., 2019)", 
                 fontsize=14, pad=15)
    
    # Limites visuais
    # Espessura de ~0.05mm a ~0.6mm (típico para condensação de água)
    ax.set_ylim(min(Delta_mm), max(Delta_mm)*1.2) 
    ax.set_xlim(0.05, 1.2)
    
    # Ajuste dos Ticks do eixo Y para mostrar números legíveis (não 10^-1)
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())

    # 7. Anotações de Engenharia
    ax.text(0.2, 0.06, "Filmes Finos\nAlta Troca Térmica", 
            ha='center', fontsize=10, style='italic', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.text(0.5, 0.4, "Filmes Espessos\n(Efeito da Turbulência visível)", 
            ha='center', fontsize=10, style='italic', color='darkred')

    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plotar_espessura_fang_vs_nu()