import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class FluidFilmProps:
    g: float = 9.81
    rho: float = 957.9
    mu: float = 2.838e-4
    k: float = 0.679
    h_fg: float = 2_257_000.0
    T_s: float = 373.15
    T_w: float = 371.15


@dataclass
class Geometry:
    L: float = 2.0
    nx: int = 300
    ny: int = 300
    x0: float = 0.01  # evita δ = 0 em x = 0


class FilmModel:
    def __init__(self, props: FluidFilmProps, geom: Geometry):
        self.p = props
        self.g = geom

    def film_thickness(self, x):
        p = self.p
        num = 4.0 * p.mu * p.k * (p.T_s - p.T_w) * x
        den = p.rho**2 * p.g * p.h_fg
        return (num / den) ** 0.25

    def velocity_profile(self, y_local, d):
        p = self.p
        factor = (p.rho * p.g) / (2.0 * p.mu)
        return factor * (2.0 * d * y_local - y_local**2)

    def average_velocity(self, d):
        p = self.p
        return p.rho * p.g * d**2 / (3.0 * p.mu)

    def reynolds_based_on_delta(self, d):
        p = self.p
        u_avg = self.average_velocity(d)
        return p.rho * u_avg * d / p.mu

    def thermal_ratio(self, Pr):
        return Pr ** (-1.0 / 3.0)

    def thermal_thickness(self, d, Pr):
        return d * self.thermal_ratio(Pr)

    def build_fields_for_pr(self, Pr):
        p = self.p
        g = self.g

        x = np.linspace(g.x0, g.L, g.nx)
        d = self.film_thickness(x)
        d_T = self.thermal_thickness(d, Pr)

        y_max = 1.2 * np.max(d_T)
        y = np.linspace(-y_max, y_max, g.ny)

        T = np.full((g.ny, g.nx), np.nan, dtype=float)
        delta_T = p.T_s - p.T_w

        for i in range(g.nx):
            dTi = d_T[i]
            if dTi <= 0:
                continue

            # interno
            mask_int = (y <= 0.0) & ((-y) <= dTi)
            if np.any(mask_int):
                yabs_int = -y[mask_int]
                T_int = p.T_w + (yabs_int / dTi) * delta_T
                T[mask_int, i] = T_int - 273.15

            # externo
            mask_ext = (y >= 0.0) & (y <= dTi)
            if np.any(mask_ext):
                yabs_ext = y[mask_ext]
                T_ext = p.T_w + (yabs_ext / dTi) * delta_T
                T[mask_ext, i] = T_ext - 273.15

        d_end = d[-1]
        dT_end = d_T[-1]
        Re_delta = self.reynolds_based_on_delta(d_end)
        Nu_L = g.L / dT_end

        X_plot, Y_plot = np.meshgrid(x, y)
        return x, y, X_plot, Y_plot, d, d_T, T, Re_delta, Nu_L

    def plot_pr_cases(self, pr_list):
        num_cases = len(pr_list)
        results = []
        t_min = np.inf
        t_max = -np.inf

        # pré-cálculo para pegar Tmin/Tmax globais
        for Pr in pr_list:
            res = self.build_fields_for_pr(Pr)
            results.append((Pr,) + res)
            T_field = res[6]
            t_min = min(t_min, np.nanmin(T_field))
            t_max = max(t_max, np.nanmax(T_field))

        fig, axes = plt.subplots(1, num_cases, figsize=(5 * num_cases, 5), sharey=True)
        if num_cases == 1:
            axes = [axes]

        cmap = plt.get_cmap('coolwarm')

        for ax, (Pr, x, y, X, Y, d, d_T, T_field, Re_delta, Nu_L) in zip(axes, results):
            pcm = ax.pcolormesh(x, 1e3 * Y, T_field, shading='auto', cmap=cmap,
                                vmin=t_min, vmax=t_max)

            film_int_mm = -1e3 * d
            film_ext_mm = 1e3 * d
            therm_int_mm = -1e3 * d_T
            therm_ext_mm = 1e3 * d_T

            ax.plot(x, film_int_mm, '--', color='black', linewidth=1.5,
                    label=r'$\delta_{v,\mathrm{int}}$')
            ax.plot(x, therm_int_mm, '-', color='cyan', linewidth=1.5,
                    label=r'$\delta_{T,\mathrm{int}}$')
            ax.plot(x, film_ext_mm, '-.', color='black', linewidth=1.5,
                    label=r'$\delta_{v,\mathrm{ext}}$')
            ax.plot(x, therm_ext_mm, '-', color='orange', linewidth=1.5,
                    label=r'$\delta_{T,\mathrm{ext}}$')

            max_x_range = x[-1] - x[0]
            profile_width_x = 0.05 * max_x_range
            x_samples = [0.5, 1.0, 1.5, 2.0]

            for x0 in x_samples:
                d0 = np.interp(x0, x, d)
                y_local = np.linspace(0.0, d0, 60)
                u = self.velocity_profile(y_local, d0)
                u_max = np.max(u)
                u_norm = u / u_max if u_max != 0 else u

                # interno
                y_prof_int = -1e3 * y_local
                x_prof_int = x0 - u_norm * profile_width_x
                ax.plot(x_prof_int, y_prof_int, color='green', linewidth=1.0)
                for idx in np.linspace(0, len(y_prof_int) - 1, 3, dtype=int):
                    yv = y_prof_int[idx]
                    xv = x_prof_int[idx]
                    ax.annotate('', xy=(xv, yv), xytext=(x0, yv),
                                arrowprops=dict(arrowstyle='->', linewidth=0.8, color='green'))

                # externo
                y_prof_ext = 1e3 * y_local
                x_prof_ext = x0 + u_norm * profile_width_x
                ax.plot(x_prof_ext, y_prof_ext, color='blue', linewidth=1.0)
                for idx in np.linspace(0, len(y_prof_ext) - 1, 3, dtype=int):
                    yv = y_prof_ext[idx]
                    xv = x_prof_ext[idx]
                    ax.annotate('', xy=(xv, yv), xytext=(x0, yv),
                                arrowprops=dict(arrowstyle='->', linewidth=0.8, color='blue'))

            ax.set_xlabel('Posição axial x (m)')
            if ax is axes[0]:
                ax.set_ylabel('Coordenada normal y (mm)\n(<0 interno, >0 externo)')
            ax.set_title(f'Pr = {Pr:.2f}')
            ax.grid(True, linestyle='--', alpha=0.4)

            texto = f'Pr = {Pr:.2f}\nRe_δ ≈ {Re_delta:,.0f}\nNu_L ≈ {Nu_L:,.1f}'
            ax.text(0.02, 0.98, texto, transform=ax.transAxes,
                    fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # colorbar única
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = plt.Normalize(t_min, t_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Temperatura (°C)')

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4,
                   bbox_to_anchor=(0.5, 1.02))

        fig.tight_layout(rect=[0.0, 0.0, 0.9, 0.95])
        plt.show()


if __name__ == '__main__':
    props = FluidFilmProps()
    geom = Geometry()
    model = FilmModel(props, geom)
    pr_values = [100.0, 1.0, 0.7]
    model.plot_pr_cases(pr_values)
