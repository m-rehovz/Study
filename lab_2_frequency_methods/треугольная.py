import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

plt.rcParams['font.size'] = 12
t = np.linspace(-5, 5, 1000)
omega = np.linspace(-20, 20, 1000)


def triangular_pulse(t, a, b):
    return np.where(np.abs(t) <= b, a * (1 - np.abs(t) / b), 0)


def fourier_triangular(omega, a, b):
    return a * b * (np.sinc(omega * b / (2 * np.pi))) ** 2  # np.sinc(x) = sin(πx)/(πx)


# --- Вариант 1: Фиксируем a=1, меняем b --- #
a_fixed = 1
b_values = [1, 2, 5]

plt.figure(figsize=(12, 5))

# Графики f(t)
plt.subplot(1, 2, 1)
for b in b_values:
    plt.plot(t, triangular_pulse(t, a_fixed, b), label=f' a = {a_fixed}, b = {b}')
plt.xlabel('$t$')
plt.ylabel('$f(t)$')
plt.legend()
plt.grid()

# Графики f̂(ω)
plt.subplot(1, 2, 2)
for b in b_values:
    plt.plot(omega, fourier_triangular(omega, a_fixed, b), label=f' a = {a_fixed}, b = {b}')
plt.xlabel('$\omega$')
plt.ylabel('$\hat{f}(\omega)$')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# --- Вариант 2: Фиксируем b=2, меняем a --- #
b_fixed = 2
a_values = [0.5, 4, 7]

plt.figure(figsize=(12, 5))

# Графики f(t)
plt.subplot(1, 2, 1)
for a in a_values:
    plt.plot(t, triangular_pulse(t, a, b_fixed), label=f'b = {b_fixed}, a = {a}')
plt.xlabel('$t$')
plt.ylabel('$f(t)$')
plt.legend()
plt.grid()

# Графики f̂(ω)
plt.subplot(1, 2, 2)
for a in a_values:
    plt.plot(omega, fourier_triangular(omega, a, b_fixed), label=f'b = {b_fixed}, a = {a}')
plt.xlabel('$\omega$')
plt.ylabel('$\hat{f}(\omega)$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Проверка равенства Парсеваля
a_test, b_test = 3, 3
energy_t = quad(lambda t: triangular_pulse(t, a_test, b_test) ** 2, -np.inf, np.inf)[0]


def integrand(omega):
    return np.abs(fourier_triangular(omega, a_test, b_test)) ** 2


energy_omega = quad(integrand, -np.inf, np.inf)[0] / (2 * np.pi)
print(f"Энергия во временной области: {energy_t:.20f}")
print(f"Энергия в частотной области: {energy_omega:.20f}")
