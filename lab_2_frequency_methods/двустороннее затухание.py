import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

plt.rcParams['font.size'] = 12
t = np.linspace(-3, 3, 1000)
omega = np.linspace(-10, 10, 1000)


def double_exp(t, a, b):
    return a * np.exp(-b * np.abs(t))


def fourier_double_exp(omega, a, b):
    return np.sqrt(2 / np.pi) * (a * b) / (b ** 2 + omega ** 2)


# --- Вариант 1: Фиксируем a=1, меняем b --- #
a_fixed = 1
b_values = [0.5, 1, 2]
plt.figure(figsize=(12, 5))

# Графики f(t)
plt.subplot(1, 2, 1)
for b in b_values:
    plt.plot(t, double_exp(t, a_fixed, b), label=f'a = {a_fixed}, b = {b}')
plt.xlabel('$t$')
plt.ylabel('$f(t)$')
plt.legend()
plt.grid()

# Графики f̂(ω)
plt.subplot(1, 2, 2)
for b in b_values:
    plt.plot(omega, fourier_double_exp(omega, a_fixed, b),
             label=f'a = {a_fixed}, b = {b}')
plt.xlabel('$\omega$')
plt.ylabel('$\hat{f}(\omega)$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# --- Вариант 2: Фиксируем b=1, меняем a --- #
b_fixed = 1
a_values = [0.5, 2, 4]

plt.figure(figsize=(12, 5))

# Графики f(t)
plt.subplot(1, 2, 1)
for a in a_values:
    plt.plot(t, double_exp(t, a, b_fixed), label=f'b = {b_fixed}, a = {a}')
plt.xlabel('$t$')
plt.ylabel('$f(t)$')
plt.legend()
plt.grid()

# Графики f̂(ω)
plt.subplot(1, 2, 2)
for a in a_values:
    plt.plot(omega, fourier_double_exp(omega, a, b_fixed),
             label=f'b = {b_fixed}, a = {a}')
plt.xlabel('$\omega$')
plt.ylabel('$\hat{f}(\omega)$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Проверка равенства Парсеваля
a_test, b_test = 1, 1
energy_t = quad(lambda t: double_exp(t, a_test, b_test) ** 2, -np.inf, np.inf)[0]
energy_omega = quad(lambda omega: np.abs(fourier_double_exp(omega, a_test, b_test)) ** 2, -np.inf, np.inf)[0]
print(f"Энергия во временной области: {energy_t:.10f}")
print(f"Энергия в частотной области: {energy_omega:.10f}")
