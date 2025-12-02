# -*- coding: utf-8 -*-
"""
SIMULAÇÃO NUMÉRICA DE TRANSFERÊNCIA DE CALOR E MASSA EM FILMES DESCENDENTES

Assunto: Análise da influência do Número de Prandtl nos perfis de camada limite.
Referência Teórica: Fang, J., Li, K., & Diao, M. (2019). "Establishment of the falling film 
evaporation model and correlation of the overall heat transfer coefficient". 
Royal Society Open Science.

Descrição:
O presente algoritmo modela os perfis de temperatura e velocidade em um tubo de evaporação/condensação.
As equações de transporte (Nusselt para hidrodinâmica e correlações empíricas para térmica) são 
resolvidas para determinar as espessuras das camadas limite (delta e delta_t) e a distribuição
de temperatura bidimensional.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Configuração de estilo gráfico para padrão de publicação
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')

def executar_simulacao_academica():
    # ==========================================================================
    # 1. PARÂMETROS DE ENTRADA (CONDIÇÕES DE CONTORNO E PROPRIEDADES)
    # ==========================================================================
    # Geometria do evaporador (Conforme Tabela 2 - Fang et al., 2019)
    L = 2.0           # Comprimento do tubo [m]
    wall_w = 0.002    # Espessura da parede (aço inoxidável) [m]
    d_i = 0.020       # Diâmetro interno [m]
    
    # Propriedades termofísicas médias (Sistema Etanol-Água)
    k_liq = 0.45      # Condutividade térmica [W/m.K]
    
    # Temperaturas de operação
    T_steam = 390.0   # Vapor saturado (Lado do Casco) [K]
    T_feed = 350.0    # Alimentação líquida (Lado do Tubo) [K]

    # Paleta de cores (Padrão técnico: Azul=Líquido, Laranja=Térmico, Cinza=Sólido)
    COR_INTERFACE = '#0055AA'
    COR_TERMICA = '#DD6600'
    COR_PAREDE = '#C0C0C0'

    # Definição dos casos de estudo baseados na difusividade (Número de Prandtl)
    # Pr = (Difusividade de Momento) / (Difusividade Térmica)
    # ATUALIZAÇÃO: Último caso alterado para Pr = 100 (Óleos muito viscosos)
    cases = [
        {'pr': 0.7,   'title': 'Gases / Ar',           'sub': r'(Pr = 0.7)'},
        {'pr': 1.0,   'title': 'Fluido Ideal / Água', 'sub': r'(Pr = 1.0)'},
        {'pr': 100.0, 'title': 'Óleos Pesados',       'sub': r'(Pr = 100.0)'}
    ]

    # ==========================================================================
    # 2. MODELAGEM MATEMÁTICA (EQUAÇÕES CONSTITUTIVAS)
    # ==========================================================================

    def calc_h_laminar(Re_l, Re_v):
        """
        Cálculo do coeficiente de filme (h_i) para regime laminar.
        Equação 3.14 (Fang et al., 2019): Nu = 62.09 * Re_l^(-0.012) * Re_v^(0.34)
        """
        Nu = 62.09 * (Re_l**-0.01239) * (Re_v**0.3427)
        h = Nu * k_liq / d_i  # h [W/m²K]
        return h

    def get_film_thickness_condensing(y_norm):
        """
        Espessura do filme de condensado (Lado Externo).
        Modelo de Nusselt (Eq. 2.7): Crescimento proporcional a y^(1/4).
        """
        base_scaling = 0.005 # Fator de escala visual
        return base_scaling * (y_norm + 0.02)**0.25

    def get_film_thickness_evaporating(y_norm):
        """
        Espessura do filme de evaporação (Lado Interno).
        Modelo simplificado de redução de espessura por transferência de massa.
        """
        base_scaling = 0.005
        return np.maximum(base_scaling * (1.0 - 0.15 * y_norm), 0)

    def get_thermal_boundary_layer(delta_h, pr):
        """
        Espessura da Camada Limite Térmica (delta_t).
        Correlação de similaridade para escoamento laminar: delta_t ~ delta_h * Pr^(-1/3).
        """
        return delta_h * (pr**(-1.0/3.0))

    def calc_temperature_profile(dist_wall, delta_t, T_wall, T_free):
        """
        Perfil transversal de temperatura T(y).
        Baseado na Figura 6: Gradiente máximo na parede, decaindo quadraticamente.
        """
        if dist_wall > delta_t:
            return T_free 
        
        eta = dist_wall / delta_t
        # Aproximação polinomial para o perfil de temperatura convectivo
        return T_wall + (T_free - T_wall) * (1.0 - (1.0 - eta)**2)

    def draw_velocity_vectors(ax, y_pos, wall_x, thick, side):
        """
        Renderização dos vetores de velocidade (Perfil Parabólico de Poiseuille).
        """
        n_vectors = 4
        scale_factor = 0.10
        eta = np.linspace(0.1, 0.9, n_vectors)
        
        # Perfil u* = 2n - n^2 (Eq. 2.13 simplificada)
        u_profile = 2*eta - eta**2
        
        direction = 1 if side == 'right' else -1
        x_coords = wall_x + (eta * thick * direction)
        
        # Linha de perfil
        ax.plot(x_coords, np.full_like(x_coords, y_pos) - u_profile*scale_factor, 
                color='black', alpha=0.3, lw=0.5)
        
        # Vetores
        for i in range(n_vectors):
            ax.annotate('', 
                        xy=(x_coords[i], y_pos - u_profile[i]*scale_factor), 
                        xytext=(x_coords[i], y_pos),
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.6, lw=0.8))

    # ==========================================================================
    # 3. ROTINA DE CÁLCULO E GERAÇÃO GRÁFICA
    # ==========================================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    
    # Malha espacial
    ny, nx = 150, 60
    y_phys = np.linspace(L, 0, ny)       # Coordenada física (m)
    y_norm = np.linspace(0, 1, ny)       # Coordenada normalizada
    w = wall_w / 2                       

    # Parâmetros de referência (Regime Laminar, conforme validação Fig. 12)
    Re_liq_max = 1200   
    Re_vap = 20000      

    for ax, case in zip(axes, cases):
        pr = case['pr']
        
        # Cálculo de h para consistência física
        h_val = calc_h_laminar(Re_liq_max, Re_vap)
        
        # Arrays de armazenamento
        X_ext, Y_ext, T_ext = [], [], []
        X_int, Y_int, T_int = [], [], []
        bounds_ext_h, bounds_ext_t = [], []
        bounds_int_h, bounds_int_t = [], []
        
        # Loop espacial (Iteração ao longo do tubo)
        for i, y in enumerate(y_phys):
            yn = y_norm[i]
            
            # --- Lado Externo ---
            delta_h_ext = get_film_thickness_condensing(yn)
            delta_t_ext = get_thermal_boundary_layer(delta_h_ext, pr)
            
            # --- Lado Interno ---
            delta_h_int = get_film_thickness_evaporating(yn)
            delta_t_int = get_thermal_boundary_layer(delta_h_int, pr)
            
            # Fronteiras
            bounds_ext_h.append(w + delta_h_ext)
            bounds_ext_t.append(w + delta_t_ext)
            bounds_int_h.append(-w - delta_h_int)
            bounds_int_t.append(-w - delta_t_int)
            
            # Campo de Temperatura (Malha 2D)
            x_mesh_ext = np.linspace(w, w + delta_h_ext, nx)
            t_profile_ext = [calc_temperature_profile(x-w, delta_t_ext, T_steam-10, T_steam) for x in x_mesh_ext]
            X_ext.append(x_mesh_ext); Y_ext.append(np.full(nx, y)); T_ext.append(t_profile_ext)
            
            x_mesh_int = np.linspace(-w - delta_h_int, -w, nx)
            t_profile_int = [calc_temperature_profile(-w-x, delta_t_int, T_feed+30, T_feed) for x in x_mesh_int]
            X_int.append(x_mesh_int); Y_int.append(np.full(nx, y)); T_int.append(t_profile_int)

        # --- Plotagem ---

        # Mapa de Calor
        cmap = 'RdYlBu_r'
        ax.pcolormesh(np.array(X_ext), np.array(Y_ext), np.array(T_ext), cmap=cmap, shading='gouraud', vmin=340, vmax=390, alpha=0.9)
        ax.pcolormesh(np.array(X_int), np.array(Y_int), np.array(T_int), cmap=cmap, shading='gouraud', vmin=340, vmax=390, alpha=0.9)
        
        # Parede do Tubo
        ax.add_patch(Rectangle((-w, 0), 2*w, L, facecolor=COR_PAREDE, edgecolor='dimgray', hatch='///', zorder=5))
        
        # Linhas de Interface
        ax.plot(bounds_ext_h, y_phys, color=COR_INTERFACE, lw=3, zorder=10)
        ax.plot(bounds_int_h, y_phys, color=COR_INTERFACE, lw=3, zorder=10)
        
        # Linhas Térmicas
        ax.plot(bounds_ext_t, y_phys, color=COR_TERMICA, linestyle='--', lw=2.5, zorder=11)
        ax.plot(bounds_int_t, y_phys, color=COR_TERMICA, linestyle='--', lw=2.5, zorder=11)
        
        # Indicação de Extrapolação Térmica (Pr < 1)
        if pr < 1.0:
            ax.fill_betweenx(y_phys, bounds_ext_h, bounds_ext_t, color=COR_TERMICA, alpha=0.15, hatch='..', zorder=9)
            ax.fill_betweenx(y_phys, bounds_int_h, bounds_int_t, color=COR_TERMICA, alpha=0.15, hatch='..', zorder=9)
            if ax == axes[0]:
                ax.annotate(r'$\delta_t$ > $\delta$', xy=(0.008, 1.8), color=COR_TERMICA, fontweight='bold', fontsize=12)

        # Vetores de Velocidade
        for ay in [1.5, 0.5]:
            idx = (np.abs(y_phys - ay)).argmin()
            draw_velocity_vectors(ax, ay, w, bounds_ext_h[idx]-w, 'right')
            draw_velocity_vectors(ax, ay, -w, abs(bounds_int_h[idx]-(-w)), 'left')

        # Formatação
        ax.set_title(f"{case['title']}\n{case['sub']}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(-0.025, 0.025)
        ax.set_xticks([-0.015, 0, 0.015])
        ax.set_xticklabels(['Evaporação\n(Interno)', '', 'Condensação\n(Externo)'], fontsize=10, color='#555')
        
        if ax == axes[0]:
            ax.set_ylabel("Altura Axial (m)", fontsize=12, fontweight='bold')
        
        ax.grid(True, linestyle=':', linewidth=1, alpha=0.5)

        # Valor de Prandtl
        ax.text(0.05, 0.95, f"Pr = {pr}", transform=ax.transAxes, 
                fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.9), zorder=20)

        # Eixo Secundário: Reynolds Local (Re_x ~ x^1.25)
        ax_re = ax.twinx()
        ax_re.set_ylim(0, L)
        y_ticks_re = np.linspace(L, 0, 6)
        re_vals = [int(Re_liq_max * (((L-y)/L)**1.25)) for y in y_ticks_re]
        
        ax_re.set_yticks(y_ticks_re)
        ax_re.set_yticklabels(re_vals, fontsize=9, color='gray')
        ax_re.set_ylabel(r"Reynolds Local ($Re_x$)", fontsize=11, color='gray', labelpad=10)
        ax_re.spines['right'].set_visible(False)
        ax_re.tick_params(axis='y', colors='gray')

    # Legenda Global
    legend_elements = [
        Line2D([0], [0], color=COR_INTERFACE, lw=3, label=r'Interface Líquida ($\delta$) - Hidrodinâmica'),
        Line2D([0], [0], color=COR_TERMICA, linestyle='--', lw=2.5, label=r'Fronteira Térmica ($\delta_t$)'),
        Rectangle((0,0),1,1, facecolor=COR_TERMICA, alpha=0.2, hatch='..', label='Difusão Térmica Dominante (Pr < 1)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12, 
               bbox_to_anchor=(0.5, 0.02), frameon=True, fancybox=True, shadow=True)

    plt.suptitle("Distribuição de Temperatura e Camadas Limite em Filmes Descendentes", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)
    
    plt.savefig("simulacao_final_pr100.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    executar_simulacao_academica()