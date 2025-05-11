import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

a, b = 1.0, 2.0
c_values = [-2, 0.5, 3]  # Сдвиг
colors = ['blue', 'green', 'red']
omega = np.linspace(-15, 15, 1000)


def shifted_triangular(t, a, b, c):
    return np.where(np.abs(t + c) <= b, a * (1 - np.abs(t + c) / b), 0)


def fourier_shifted_triangular(omega_val, a, b, c):
    if np.isclose(omega_val, 0, atol=1e-10):
        return (a * b) / np.sqrt(2 * np.pi)
    else:
        return (4 * a * np.sin(omega_val * b / 2) ** 2) / (b * omega_val ** 2 * np.sqrt(2 * np.pi)) * np.exp(
            1j * omega_val * c)


plt.figure(figsize=(14, 6))

# Сдвинутые функции
plt.subplot(1, 2, 1)
t = np.linspace(-5, 5, 1000)
for c, color in zip(c_values, colors):
    plt.plot(t, shifted_triangular(t, a, b, c), color=color, label=f'c = {c}')
plt.title('Сдвинутые функции g(t)')
plt.xlabel('t')
plt.ylabel('g(t)')
plt.legend()
plt.grid()

# Модули спектров
plt.subplot(1, 2, 2)
linestyles = ['--', '-.', ':']
for c, color, linestyle in zip(c_values, colors, linestyles):
    F = np.array([fourier_shifted_triangular(w, a, b, c) for w in omega])
    plt.plot(omega, np.abs(F), color=color, linestyle=linestyle, label=f'c = {c}')
plt.title('Модули Фурье-образов |ĝ(ω)|')
plt.xlabel('ω')
plt.ylabel('ĝ(ω)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Графики вещественной и мнимой частей
for c in c_values:
    plt.figure(figsize=(12, 5))
    F = np.array([fourier_shifted_triangular(w, a, b, c) for w in omega])

    plt.plot(omega, np.real(F), 'b-', label='Вещественная часть')
    plt.plot(omega, np.imag(F), 'r--', label='Мнимая часть')

    plt.title(f'Фурье-образ для c = {c}')
    plt.xlabel('ω')
    plt.ylabel('ĝ(ω)')
    plt.legend()
    plt.grid()
    plt.show()

print("\nПроверка равенства Парсеваля:")
# Аналитические значения
energy_analytic = (2 * a ** 2 * b) / 3


# Численное интегрирование во временной области
def time_integrand(t, c=0):
    return shifted_triangular(t, a, b, c) ** 2


energy_time, _ = quad(time_integrand, -b, b)  # Функция отлична от нуля только на [-b,b]


# Численное интегрирование в частотной области
def freq_integrand(omega_val, c=0):
    F = fourier_shifted_triangular(omega_val, a, b, c)
    return np.abs(F) ** 2


# Разбиваем интеграл на две части для лучшей точности
energy_freq1, _ = quad(freq_integrand, -100, 0, args=(0,))
energy_freq2, _ = quad(freq_integrand, 0, 100, args=(0,))
energy_freq = energy_freq1 + energy_freq2
print(f"Аналитическое значение энергии: {energy_analytic:.6f}")
print(f"Численное значение (временная область): {energy_time:.6f}")
print(f"Численное значение (частотная область): {energy_freq:.6f}")
print(f"Разница (время/аналитика): {abs(energy_time - energy_analytic):.2e}")
print(f"Разница (частота/аналитика): {abs(energy_freq - energy_analytic):.2e}")
