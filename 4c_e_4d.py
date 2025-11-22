import numpy as np
import matplotlib.pyplot as plt

# --- Constantes e Parâmetros ---
# Valores de Re para a faixa de escoamento (100 a 10000)
RE_RANGE = np.logspace(2, 4, 100) 
# Valores de Pr para diferentes fluidos (Metais, Gases, Água/Líquidos)
PR_VALUES = [0.01, 1.0, 10.0] 
# Constante na correlação Nu (62.09 * (di/kl) - ajustada para visualização)
NU_CONSTANT = 62.09
# Diâmetro interno do tubo (Exemplo, 20mm conforme a Tabela 2)
DIAMETER = 0.020 

# --- Funções de Correlação ---

def calculate_nu(Re, Pr, d_i=DIAMETER, k_l=0.679):
    """
    Calcula o Número de Nusselt (Nu) para Escoamento Laminar Interno (Evaporação).
    Baseado na forma da Eq. 3.14 e 3.16.
    
    Nu = C * Pr^0.5427 * Re^0.0178 * (di/kl)
    O termo d_i/k_l é incluído na constante 62.09 na forma simplificada (Eq. 3.16),
    mas é reintroduzido para manter a forma original e flexibilidade.
    """
    # Usamos 1.0 para o termo (d_i/k_l) para plotar apenas a dependência Re-Pr
    # A constante 62.09 absorve o d_i/k_l de alguma forma para dar o coeficiente
    nu = 62.09 * (Pr ** 0.5427) * (Re ** 0.0178) 
    return nu

def calculate_delta(Re, d_i=DIAMETER):
    """
    Calcula a espessura do filme (delta) com base no Re, 
    usando a correlação da subcamada laminar (Eq. 2.21, simplificada da 2.19, 
    mostrada como δ/d = 61.5 Re^(-7/8)).
    
    Retorna delta em metros (m).
    """
    # delta / d_i = 61.5 * Re^(-7/8)
    delta_over_di = 61.5 * (Re ** (-7.0 / 8.0))
    delta = delta_over_di * d_i
    return delta

# --- Item 4C: Nu vs Re para diferentes Pr (Gráfico de Contorno) ---

def plot_nu_vs_re():
    plt.figure(figsize=(8, 6))
    
    for Pr in PR_VALUES:
        Nu = calculate_nu(RE_RANGE, Pr)
        
        # Plotar no eixo log-log para melhor visualização da relação de potência
        plt.loglog(RE_RANGE, Nu, marker='o', markersize=4, linestyle='-', label=f'Pr = {Pr:.2f}')

    plt.xlabel('Número de Reynolds (Re)')
    plt.ylabel('Número de Nusselt (Nu)')
    plt.title(r'Item 4C: $Nu$ vs $Re$ para diferentes $Pr$ (Evaporação Laminar)')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    
    # Análise da Correlação
    plt.text(0.05, 0.95, 
             r'$\mathbf{Nu} \propto \mathbf{Pr}^{0.54} \mathbf{Re}^{0.018}$', 
             transform=plt.gca().transAxes, fontsize=12, va='top', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

    plt.legend(title='Número de Prandtl', loc='lower right')
    plt.show()

# --- Item 4D: Nu vs Espessura do Filme (delta) ---

def plot_nu_vs_delta():
    plt.figure(figsize=(8, 6))
    
    # Escolhemos um Pr (Exemplo: Pr=10, Água)
    Pr_ref = 10.0
    
    # 1. Calcular Delta e Nu para a faixa de Re
    Delta_m = calculate_delta(RE_RANGE)
    Nu = calculate_nu(RE_RANGE, Pr_ref)
    
    # Converter Delta para micrometros (µm) para melhor escala no gráfico
    Delta_um = Delta_m * 1e6 
    
    plt.plot(Delta_um, Nu, 'k-', linewidth=2)
    
    plt.xlabel(r'Espessura do Filme, $\delta$ ($\mu$m)')
    plt.ylabel('Número de Nusselt (Nu)')
    plt.title(r'Item 4D: $Nu$ vs Espessura do Filme ($\delta$) para $Pr=10$')
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Análise da Correlação (Nu ∝ delta^-0.0203)
    plt.text(0.05, 0.95, 
             r'$\mathbf{Nu} \propto \mathbf{\delta}^{-0.02} \mathbf{Pr}^{0.54}$', 
             transform=plt.gca().transAxes, fontsize=12, va='top', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

    plt.show()

if __name__ == '__main__':
    print("Gerando gráfico para o Item 4C...")
    plot_nu_vs_re()
    print("Gerando gráfico para o Item 4D...")
    plot_nu_vs_delta()