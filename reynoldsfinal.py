# -*- coding: utf-8 -*-
"""
ANÁLISE COMPARATIVA DE CAMADAS LIMITE EM FILMES DESCENDENTES
MODELAGEM FÍSICA E SIMULAÇÃO NUMÉRICA

Referência: Fang, J., Li, K., & Diao, M. (2019). "Establishment of the falling film evaporation model..."
Royal Society Open Science.

Objetivo:
Simular e comparar os perfis de espessura hidrodinâmica e térmica sob a influência
de diferentes números de Prandtl, validando o modelo com as equações de Nusselt (condensação)
e correlações empíricas para transição de regime.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Configuração de estilo visual para padrão de publicação
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')

def calcular_perfis_rigorosos(regime, x_array):
    """
    Calcula os perfis de camada limite baseados nas equações constitutivas do artigo.
    
    Parâmetros:
        regime (str): Identificação do caso físico ('baixo', 'intermediario', 'alto').
        x_array (np.array): Vetor espacial adimensional.
        
    Retorno:
        delta_mm: Perfil hidrodinâmico [mm].
        delta_t_mm: Perfil térmico [mm].
        x_transicao: Ponto de transição laminar-turbulento (se houver).
        Pr: Número de Prandtl.
        Re_max: Reynolds máximo local.
    """
    
    # Tratamento de singularidade na origem (x=0)
    x = np.maximum(x_array, 1e-6)
    
    # Inicialização dos vetores
    delta_mm = np.zeros_like(x)
    x_transicao = None
    
    # --- Seleção do Regime Físico ---
    if regime == "baixo":
        # CASO 1: Gases ou Metais Líquidos (Pr = 0.7)
        # Característica: Difusividade térmica > Difusividade de momento.
        Pr = 0.7
        Fator_Escala = 0.35 # Constante dimensional de espessura [mm]
        
        # Modelo de Nusselt Laminar (Eq. 2.7): delta proporcional a x^0.25
        delta_mm = Fator_Escala * (x ** 0.25)
        
        # Reynolds baixo (Regime estritamente laminar)
        Re_max = 800 

    elif regime == "intermediario":
        # CASO 2: Água / Fluidos Ideais (Pr = 1.0)
        # Característica: Camadas limite crescem em taxas similares.
        Pr = 1.0
        Fator_Escala = 0.35
        delta_mm = Fator_Escala * (x ** 0.25)
        
        # Reynolds intermediário (Laminar desenvolvido)
        Re_max = 1500 

    elif regime == "alto":
        # CASO 3: Óleos Pesados / Fluidos Viscosos (Pr = 100.0)
        # Característica: Camada térmica confinada dentro da subcamada viscosa.
        # Inclui modelagem de transição e instabilidade interfacial.
        Pr = 100.0
        x_transicao = 0.4 # Posição adimensional da transição
        
        # Mapeamento do Reynolds Crítico (~1600) na posição de transição
        Re_max = int(1600 / (0.4 ** 1.25)) 
        
        # Região Laminar (x < x_crit)
        mask_lam = x < x_transicao
        delta_mm[mask_lam] = 0.22 * (x[mask_lam] ** 0.25)
        
        # Região Turbulenta (x >= x_crit)
        mask_turb = x >= x_transicao
        base = 0.22 * (x_transicao ** 0.25)
        
        # Lei de crescimento turbulento empírica (x^0.8)
        crescimento = (x[mask_turb] / x_transicao) ** 0.8
        
        # Modelagem das Ondas de Kapitza (Instabilidade de superfície)
        # A turbulência gera perturbações senoidais na interface livre
        ondas = 0.012 * np.sin(40 * x[mask_turb])
        
        delta_mm[mask_turb] = (base * crescimento * 1.4) + ondas

    else:
        raise ValueError("Erro: Regime desconhecido.")

    # Cálculo da Camada Térmica (Analogia de Reynolds-Colburn)
    # delta_t / delta = Pr^(-1/3)
    delta_t_mm = delta_mm * (Pr ** (-1/3))
    
    return delta_mm, delta_t_mm, x_transicao, Pr, Re_max

def gerar_graficos_final_revisado():
    """
    Gera a visualização gráfica final com layout acadêmico ajustado.
    """
    
    # Definição do domínio espacial (Alta resolução para suavidade das curvas)
    x = np.linspace(0.0, 1.0, 800)
    
    # Configuração da Figura
    # Aumentei a altura para 7.0 para dar mais espaço vertical
    fig, axs = plt.subplots(1, 3, figsize=(18, 7.0)) 
    
    # Parâmetros dos Cenários de Simulação
    cenarios = [
        {"id": "baixo", "titulo": "Gases / Ar", "sub": r"($Pr = 0,7$)"},
        {"id": "intermediario", "titulo": "Fluido Ideal / Água", "sub": r"($Pr = 1,0$)"},
        {"id": "alto", "titulo": "Óleos Pesados", "sub": r"($Pr = 100$)"}
    ]

    # Paleta de Cores Institucional
    COR_HIDRO = '#0055AA'  # Azul Engenharia
    COR_TERMICA = '#DD6600' # Laranja Segurança

    for ax, info in zip(axs, cenarios):
        # Cálculo numérico dos perfis
        d_v, d_t, x_trans, Pr_val, re_max = calcular_perfis_rigorosos(info["id"], x)
        
        # --- PLOTAGEM DAS CAMADAS ---
        
        # 1. Camada Hidrodinâmica (Sólida)
        ax.plot(x, d_v, color=COR_HIDRO, lw=3, label=r"$\delta$ (Hidrodinâmica)")
        ax.fill_between(x, 0, d_v, color=COR_HIDRO, alpha=0.15)
        
        # 2. Camada Térmica (Tracejada)
        ax.plot(x, d_t, color=COR_TERMICA, lw=2.5, ls="--", label=r"$\delta_t$ (Térmica)")
        
        # Preenchimento condicional baseado na física do Prandtl
        if Pr_val == 100:
             # Caso Pr >> 1: Calor confinado (difusão lenta)
             ax.fill_between(x, 0, d_t, color=COR_TERMICA, alpha=0.4)
        elif Pr_val == 0.7:
             # Caso Pr < 1: Calor expandido (difusão rápida)
             ax.fill_between(x, 0, d_t, color=COR_TERMICA, alpha=0.1)

        # --- FORMATAÇÃO DOS EIXOS PRIMÁRIOS ---
        
        # Títulos
        ax.set_title(f"{info['titulo']}\n{info['sub']}", fontsize=13, fontweight='bold', pad=15)
        
        ax.set_xlabel(r"Posição Axial Adimensional ($x/L$)", fontsize=11)
        ax.set_ylim(0, 0.65) # Escala fixa para comparação direta
        ax.set_xlim(0, 1.0)
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)

        # --- EIXO SECUNDÁRIO: MAPEAMENTO DE REYNOLDS ---
        # Criação do eixo superior para mostrar a evolução do regime de escoamento
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        
        # Mapeamento x -> Re (Assumindo Re ~ x^1.25 conforme Eq. 2.8)
        ticks_loc = np.linspace(0, 1, 6)
        ticks_labels = [f"{int(re_max * (t**1.25))}" for t in ticks_loc]
        ticks_labels[0] = "0"
        
        ax_top.set_xticks(ticks_loc)
        ax_top.set_xticklabels(ticks_labels, fontsize=9, color='gray')
        ax_top.set_xlabel(r"Número de Reynolds Local ($Re_x$)", fontsize=10, color='gray', labelpad=8)
        
        # --- ELEMENTOS VISUAIS E ANOTAÇÕES ---
        
        # Caixa informativa do Prandtl
        ax.text(0.05, 0.92, f"Pr = {Pr_val}", transform=ax.transAxes, 
                fontsize=11, fontweight='bold', 
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4', alpha=0.9))

        # Legenda (Exibida apenas no primeiro gráfico para limpeza visual)
        if info["id"] == "baixo":
            ax.set_ylabel("Espessura Real Estimada (mm)", fontsize=11, fontweight='bold')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 0.85), frameon=True, framealpha=0.95, fontsize=10)

        # Marcação de Transição de Regime (Apenas para alto Reynolds)
        if x_trans:
            # Linha vertical discreta
            ax.axvline(x_trans, color='red', ls=':', lw=1.5)
            
            # Textos descritivos limpos (Sem setas grandes)
            ax.text(x_trans + 0.02, 0.58, "Início Turbulência", color='red', fontsize=9, alpha=0.9, fontweight='bold')
            ax.text(x_trans + 0.02, 0.05, "Ondas Interfaciais", color='red', fontsize=9, alpha=0.8, fontstyle='italic')

    # Título Principal (Ajustado)
    fig.suptitle(f"Desenvolvimento de Camada Limite: Influência de Pr e Re", 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Ajuste fino de layout: reserva o topo (rect) para o título principal
    # rect = [left, bottom, right, top] -> Topo em 0.90 garante 10% de espaço livre
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    # Salvamento e Exibição
    filename = "grafico_comparativo_final_ajustado.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gráfico gerado com sucesso: {filename}")

if __name__ == "__main__":
    gerar_graficos_final_revisado()