# -*- coding: utf-8 -*-
"""
PARTE FINAL - MAPA DE DESEMPENHO AMPLIADO (Gases a Óleos Viscosos)
Faixa de Prandtl: 0.5 a 100

Descrição:
Gera um mapa de contorno relacionando Reynolds e Nusselt.
Usa escala logarítmica nas cores para visualizar desde Pr < 1 até Pr = 100.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # Importante para escalas grandes

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')

def calcular_mapa_prandtl_range_alto():
    # 1. Definir o Grid (Expandido no Eixo Y para caber Pr=100)
    re_vals = np.logspace(2, 4.3, 300)        # Re: 100 a 20.000
    # Aumentamos o Nu máximo para 5.0 para capturar a alta troca térmica de Pr=100
    nu_vals = np.logspace(np.log10(0.05), np.log10(5.0), 300) 
    
    Re, Nu = np.meshgrid(re_vals, nu_vals)
    
    # 2. Matemática Inversa: Dado Re e Nu, qual é o Pr?
    Nu_lam = 1.10 * (Re ** (-1/3))
    
    # Inicializa matriz com NaN
    Pr_map = np.full_like(Re, np.nan)
    
    # Máscara: Só calculamos onde Nu > Nu_lam (Fisicamente possível)
    mask = Nu > Nu_lam
    
    # Pr = (Nu_turb / (0.0038 * Re^0.4)) ^ (1/0.65)
    Nu_turb_sq = (Nu[mask]**2) - (Nu_lam[mask]**2)
    Nu_turb_req = np.sqrt(Nu_turb_sq)
    term_re = 0.0038 * (Re[mask] ** 0.4)
    
    Pr_calc = (Nu_turb_req / term_re) ** (1/0.65)
    Pr_map[mask] = Pr_calc

    # 3. Plotagem
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # --- CAMADA 1: CORES (ESCALA LOGARÍTMICA) ---
    # Usamos LogNorm para que Pr=1 e Pr=100 sejam ambos visíveis na escala de cor
    # vmin=0.5 pega Pr<1 (gases) e vmax=100 pega os óleos.
    # cmap='Spectral_r' ou 'turbo' ou 'viridis'. 'Spectral_r' dá um contraste bonito (Azul=Baixo, Vermelho=Alto)
    cp_fill = ax.contourf(Re, Nu, Pr_map, levels=np.logspace(np.log10(0.5), np.log10(150), 100), 
                          cmap='Spectral_r', norm=LogNorm(vmin=0.5, vmax=150))
    
    cbar = fig.colorbar(cp_fill, ax=ax, pad=0.02)
    cbar.set_label(r'Número de Prandtl ($Pr$) - Escala Log', fontsize=12, fontweight='bold', labelpad=10)
    
    # Ajustar ticks da colorbar para mostrar valores chave
    cbar.set_ticks([0.7, 1, 5, 10, 50, 100])
    cbar.set_ticklabels(['0.7', '1', '5', '10', '50', '100'])

    # --- CAMADA 2: LINHAS DE CONTORNO (ESPECÍFICAS) ---
    # Definimos exatamente as linhas que queremos ver
    levels_lines = [0.7, 1, 2, 5, 10, 20, 50, 100]
    cp_lines = ax.contour(Re, Nu, Pr_map, levels=levels_lines, colors='black', linewidths=0.8, linestyles='-')
    
    # Labels nas linhas (CORRIGIDO: sem fontweight)
    # fmt personaliza para não ficar cheio de zeros decimais
    ax.clabel(cp_lines, inline=True, fontsize=9, fmt='%1.3g', colors='black')

    # --- CAMADA 3: LIMITE LAMINAR ---
    re_line = np.linspace(100, 20000, 300)
    nu_limite = 1.10 * (re_line ** (-1/3))
    ax.plot(re_line, nu_limite, color='black', lw=2.5, linestyle='--', label='Limite Laminar (Pr → 0)')
    
    # Preencher região impossível
    ax.fill_between(re_line, 0.01, nu_limite, color='gray', alpha=0.3)
    ax.text(200, 0.06, "Região Impossível\n(Abaixo do Limite Laminar)", color='gray', fontsize=10, fontstyle='italic')

    # 4. Formatação Final
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Limites para focar onde importa
    ax.set_ylim(0.05, 5.0) 
    
    ax.set_xlabel(r'Número de Reynolds ($Re$)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Número de Nusselt ($Nu^*$)', fontsize=12, fontweight='bold')
    
    ax.set_title("Mapa Completo de Operação (Gases a Óleos)\nInteração Reynolds vs Nusselt vs Prandtl", 
                 fontsize=14, pad=15)
    
    # Pequenas anotações sobre os fluidos
    ax.text(12000, 3.5, "Óleos Viscosos\n(Pr ≈ 100)", color='darkred', ha='center', fontsize=9, fontweight='bold')
    ax.text(15000, 0.35, "Gases/Ar\n(Pr ≈ 0.7)", color='navy', ha='center', fontsize=9, fontweight='bold')

    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    calcular_mapa_prandtl_range_alto()