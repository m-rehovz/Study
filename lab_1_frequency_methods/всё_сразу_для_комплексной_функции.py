import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Параметры
R = 1
T = 2
N_max = 2

def f(t):
    t_mod = t % T # Нормализация времени в интервал [0, T)
    # Вычисление действительной части
    if 0 <= t_mod < T / 8:
        Re = R
        Im = (8 * R / T) * t_mod
    elif T / 8 <= t_mod < 3 * T / 8:
        Re = 2 * R - (8 * R / T) * t_mod
        Im = R
    elif 3 * T / 8 <= t_mod < 5 * T / 8:
        Re = -R
        Im = 4 * R - (8 * R / T) * t_mod
    elif 5 * T / 8 <= t_mod < 7 * T / 8:
        Re = -6 * R + (8 * R / T) * t_mod
        Im = -R
    else:
        Re = R  # Для 7T/8 <= t_mod < T
        Im = (8 * R / T) * (t_mod - T)  # Для 7T/8 <= t_mod < T

    return complex(Re, Im)


# Вычисление коэффициентов Фурье
def compute_cn(n):
    omega_n = 2 * np.pi * n / T
    integrand = lambda t: f(t) * np.exp(-1j * omega_n * t)
    real_part = quad(lambda t: np.real(integrand(t)), -T/8, 7*T/8, epsabs=1e-12)[0]
    imag_part = quad(lambda t: np.imag(integrand(t)), -T/8, 7*T/8, epsabs=1e-12)[0]
    return (real_part + 1j * imag_part) / T

N = N_max

coeffs = [(n, compute_cn(n)) for n in range(-N, N+1)]
def complex_sum(t, coeffs):
    sum_terms = 0.0
    for n, cn in coeffs:
        omega_n = 2 * np.pi * n / T
        sum_terms += cn * np.exp(1j * omega_n * t)
    return sum_terms


print("\nКомплексные коэффициенты:")
for n, cn in coeffs:
    print(f"c[{n:2d}] = {cn.real:+.6f} {cn.imag:+.6f}j")


def complex_sum(t, coeffs):
    return sum(cn * np.exp(1j * 2 * np.pi * n * t / T) for n, cn in coeffs)


t_values = np.linspace(0, T, 1000)
f_values = np.array([f(t) for t in t_values])
GN_values = np.array([complex_sum(t, coeffs) for t in t_values])


power_original = (1/T) * np.trapz([abs(f(t))**2 for t in t_values], t_values)
power_GN = sum(abs(cn)**2 for _, cn in coeffs)

print("\nПроверка равенства Парсеваля:")
print(f"Оригинал: {power_original:.6f}")
print(f"Ряд Фурье: {power_GN:.6f}")
print(f"Разница: {abs(power_original - power_GN):.6f}")

# Построение параметрического графика
plt.figure(figsize=(10, 10))
plt.plot(np.real(f_values), np.imag(f_values), 'b', label='Исходная функция f(t)', alpha=0.7)
plt.plot(np.real(GN_values), np.imag(GN_values), 'r--', label=f'Частичная сумма G_N(t), N={N}', linewidth=1.5)
plt.xlabel('Re(f(t))')
plt.ylabel('Im(f(t))')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Построение графиков Re и Im для сравнения
plt.figure(figsize=(12, 6))

# График для Re и Im
plt.subplot(2, 1, 1)
plt.plot(t_values, np.real(f_values), 'b', label='Re(f(t))')
plt.plot(t_values, np.imag(f_values), 'g', label='Im(f(t))')
plt.plot(t_values, np.real(GN_values), 'r--', label=f'Re(G_N(t)), N={N}')
plt.plot(t_values, np.imag(GN_values), 'm--', label=f'Im(G_N(t)), N={N}')
plt.xlabel('t')
plt.ylabel('Re, Im')
plt.title('Re и Im')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()