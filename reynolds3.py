import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --- Classes de Propriedades e Geometria ---
@dataclass
class FluidFilmProps:
    g: float = 9.81
    rho: float = 957.9
    mu: float = 2.838e-4
    k: float = 0.679
    h_fg: float = 2_257_000.0
    T_s: float = 373.15
    T_w: float = 371.15
    Cp: float = 4180.0

@dataclass
class Geometry:
    L: float = 2.0
    nx: int = 300
    ny: int = 300
    x0: float = 0.01
    
# Parâmetro de referência
RE_REF_LAMINAR: float = 4.0 # Re de referência onde o modelo de Nusselt é exato.

class FilmModel:
    def __init__(self, props: FluidFilmProps, geom: Geometry):
        self.p = props
        self.g = geom

    def film_thickness_nusselt(self, x):
        """Espessura de filme pela correlação de Nusselt (solução laminar)."""
        p = self.p
        num = 4.0 * p.mu * p.k * (p.T_s - p.T_w) * x
        den = p.rho**2 * p.g * p.h_fg
        return (num / den) ** 0.25
    
    def re_thinning_factor(self, Re_target: float, exponent: float = -0.33):
        """
        Fator de correção para simular o afinamento da espessura do filme (δ)
        quando Re aumenta. Usa uma relação de potência δ ∝ Re^exponent.
        (O expoente -0.33, ou -1/3, é comum para condensação laminar).
        
        Retorna: Fator F, onde δ_ajustado = δ_Nusselt * F
        """
        if Re_target < RE_REF_LAMINAR:
            return 1.0
            
        # Ajuste a espessura relativa à espessura de referência RE_REF_LAMINAR
        return (Re_target / RE_REF_LAMINAR) ** exponent

    def thermal_ratio(self, Pr):
        """Relação da espessura térmica: δ_T/δ = Pr^(-1/3)."""
        return Pr ** (-1.0 / 3.0)

    def thermal_thickness(self, d, Pr):
        """Espessura da camada limite térmica."""
        return d * self.thermal_ratio(Pr)

    def velocity_shape(self, Re, y_local, d):
        """
        Perfil de velocidade ADIMENSIONAL (u/u_max) em função do regime Re.
        eta = y_local / d
        """
        if d <= 0:
            return np.zeros_like(y_local)

        eta = y_local / d

        # Regimes Re: 1 (Laminar), 1000 (Transição), 10000 (Turbulento)
        if Re <= 10.0:
            u_norm = 2.0 * eta - eta**2 # Laminar (Parabólico)
        elif Re <= 3000.0:
            u_norm = eta ** (1.0 / 5.0) # Transição (Lei de Potência 1/5, valor entre 1/7 e 1/3)
        else:
            u_norm = eta ** (1.0 / 7.0) # Turbulento (Lei de Potência 1/7)

        u_norm[u_norm < 0] = 0.0
        return u_norm

    def build_temperature_field(self, Pr=1.0):
        """
        Calcula o campo de temperatura. A espessura térmica d_T é independente
        do Re, mas depende de x e Pr.
        """
        p, g = self.p, self.g
        x = np.linspace(g.x0, g.L, g.nx)
        d_nusselt = self.film_thickness_nusselt(x)
        d_T = self.thermal_thickness(d_nusselt, Pr)

        y_max = 1.2 * np.max(d_T)
        y = np.linspace(-y_max, y_max, g.ny)
        T = np.full((g.ny, g.nx), np.nan, dtype=float)
        delta_T = p.T_s - p.T_w

        for i in range(g.nx):
            dTi = d_T[i]
            # Assumimos que o campo de T se estende simetricamente para fins de ilustração
            mask = (np.abs(y) <= dTi)
            if np.any(mask):
                yabs = np.abs(y[mask])
                # T em °C
                T[mask, i] = p.T_w + (yabs / dTi) * delta_T - 273.15 

        return x, y, d_nusselt, d_T, T

    def plot_reynolds_transition(self, Re_list, Pr=10.0):
        
        x, y, d_nusselt, d_T, T = self.build_temperature_field(Pr)
        t_min = np.nanmin(T)
        t_max = np.nanmax(T)

        fig, ax = plt.subplots(figsize=(12, 6))
        cmap = plt.get_cmap('coolwarm')
        
        # Plot do campo de Temperatura (que só depende de δ_T e, portanto, de Pr e x)
        pcm = ax.pcolormesh(x, 1e3 * y, T, shading='auto', cmap=cmap,
                            vmin=t_min, vmax=t_max)
        ax.axhline(0, color='gray', linestyle='-', linewidth=2) # Linha da parede

        # Plotar Linhas de Espessura Térmica (δT)
        therm_int_mm = -1e3 * d_T
        therm_ext_mm = 1e3 * d_T

        ax.plot(x, therm_int_mm, '-', color='cyan', linewidth=1.5,
                label=r'$\delta_{T}$ (Camada Limite Térmica)')
        ax.plot(x, therm_ext_mm, '-', color='orange', linewidth=1.5,
                label=r'$\delta_{T}$')
        
        # Cores e Legendas para os Perfis de Velocidade (Re)
        colour_map = {
            Re_list[0]: 'green',  # Laminar (Baixo Re)
            Re_list[1]: 'blue',   # Transição (Médio Re)
            Re_list[2]: 'red'     # Turbulento (Alto Re)
        }

        x_samples = [0.25, 0.75, 1.25, 1.75]
        max_x_range = x[-1] - x[0]
        profile_width_x = 0.05 * max_x_range

        # Plotar Perfis de Velocidade (com δ ajustado)
        for Re in Re_list:
            color = colour_map.get(Re, 'black')
            
            # --- CÁLCULO DO FATOR DE AFINAMENTO ---
            thinning_factor = self.re_thinning_factor(Re, exponent=-0.33)
            # A espessura do filme para este Re é ajustada: δ_Re = δ_Nusselt * Fator
            d_adjusted = d_nusselt * thinning_factor
            
            # Plotar a linha da espessura δ_v para este Re específico
            ax.plot(x, -1e3 * d_adjusted, '--', color=color, linewidth=1.0, alpha=0.8,
                    label=f'$Re = {int(Re):,}$: $\delta_v$ ajustada')
            ax.plot(x, 1e3 * d_adjusted, '--', color=color, linewidth=1.0, alpha=0.8)

            for x0 in x_samples:
                # Interpolar a espessura ajustada δ(x)
                d0_adj = np.interp(x0, x, d_adjusted) 
                y_local = np.linspace(0.0, d0_adj, 60)
                u_norm = self.velocity_shape(Re, y_local, d0_adj)

                # Interno (y negativo)
                y_prof_int = -1e3 * y_local
                x_prof_int = x0 - u_norm * profile_width_x
                ax.plot(x_prof_int, y_prof_int, color=color, linewidth=1.0, alpha=0.8)
                
                # Externo (y positivo)
                y_prof_ext = 1e3 * y_local
                x_prof_ext = x0 + u_norm * profile_width_x
                ax.plot(x_prof_ext, y_prof_ext, color=color, linewidth=1.0, alpha=0.8)

        # --- Configurações do Gráfico ---
        ax.set_xlabel('Posição axial x (m)')
        ax.set_ylabel('Coordenada normal y (mm)\n(<0 interno, >0 externo)')
        ax.set_title(f'Perfis de Velocidade vs. Regime Reynolds (Pr={Pr:.1f}) com $\delta$ Ajustado')
        ax.grid(True, linestyle='--', alpha=0.4)

        # colorbar
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label('Temperatura (°C)')
        
        # Legenda Final
        lines, labels = ax.get_legend_handles_labels()
        # Remove labels duplicados (perfil de velocidade e linha de espessura de Re)
        unique_labels = {}
        for line, label in zip(lines, labels):
            if label not in unique_labels:
                unique_labels[label] = line
        
        fig.legend(unique_labels.values(), unique_labels.keys(), 
                   loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))

        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])
        plt.show()


if __name__ == '__main__':
    props = FluidFilmProps()
    geom = Geometry()
    model = FilmModel(props, geom)
    # Requerimentos: Re=1, Re>>1 (10000), Re<<1 (1)
    reynolds_values = [1.0, 1000.0, 10000.0] 
    model.plot_reynolds_transition(reynolds_values, Pr=1.0)