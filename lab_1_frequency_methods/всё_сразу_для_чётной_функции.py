import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

N = 2
T = 2 * np.pi / 1.5
t0 = 0


def compute_fourier_coefficients(N):
    results = {'a': [], 'b': [], 'c': []}

    a0 = (1 / T) * quad(lambda t: np.abs(3 * np.cos(1.5 * t)), t0, t0 + T)[0]
    results['a'].append(a0)

    for n in range(1, N + 1):
        omega_n = 2 * np.pi * n / T

        an = (2 / T) * quad(lambda t: np.abs(3 * np.cos(1.5 * t)) * np.cos(omega_n * t), t0, t0 + T)[0]
        results['a'].append(an)

        bn = (2 / T) * quad(lambda t: np.abs(3 * np.cos(1.5 * t)) * np.sin(omega_n * t), t0, t0 + T)[0]
        results['b'].append(bn)

    # cn
    for n in range(-N, N + 1):
        omega_n = 2 * np.pi * n / T
        cn = (1 / T) * quad(lambda t: np.abs(3 * np.cos(1.5 * t)) * np.exp(-1j * omega_n * t), t0, t0 + T)[0]
        results['c'].append((n, cn))

    return results


def original_function(t):
    return np.abs(3 * np.cos(1.5 * t))


def trigonometric_sum(t, N, coeffs):
    sum_terms = coeffs['a'][0]
    for n in range(1, N + 1):
        omega_n = 2 * np.pi * n / T
        sum_terms += coeffs['a'][n] * np.cos(omega_n * t) + coeffs['b'][n - 1] * np.sin(omega_n * t)
    return sum_terms


def complex_sum(t, N, coeffs):
    sum_terms = 0
    for n, cn in coeffs['c']:
        if abs(n) > N:
            continue
        omega_n = 2 * np.pi * n / T
        sum_terms += cn * np.exp(1j * omega_n * t)
    return sum_terms.real


coeffs = compute_fourier_coefficients(N)

print("Вещественные коэффициенты:")
print(f"a0 = {coeffs['a'][0]:.6f}")
for n in range(1, N + 1):
    print(f"a{n} = {coeffs['a'][n]:.6f}, b{n} = {coeffs['b'][n - 1]:.6f}")
print("\nКомплексные коэффициенты:")  # Вывод всех c_n для n от -N до N
for n, cn in coeffs['c']:
    print(f"c[{n:2d}] = {cn.real:+.6f} {cn.imag:+.6f}j")

# Проверка равенства Парсеваля
power_original = (1 / T) * quad(lambda t: np.abs(3 * np.cos(1.5 * t)) ** 2, t0, t0 + T)[0]
power_real = (coeffs['a'][0] ** 2) + 0.5 * sum(a ** 2 + b ** 2 for a, b in zip(coeffs['a'][1:], coeffs['b']))
power_complex = sum(abs(cn) ** 2 for n, cn in coeffs['c'])

print("\nПроверка равенства Парсеваля:")
print(f"Левая часть (средняя мощность сигнала): {power_original:.6f}")
print(f"Правая часть (вещественные коэффициенты): {power_real:.6f}")
print(f"Правая часть (комплексные коэффициенты): {power_complex:.6f}")
print(f"Разница (для вещественных): {abs(power_original - power_real):.6f}")
print(f"Разница (для комплексных): {abs(power_original - power_complex):.6f}")

subscript_numbers = {1: '₁', 2: '₂', 3: '₃', 4: '₄', 5: '₅', 6: '₆', 7: '₇', 8: '₈', 9: '₉', 10: '₁₀', 11: '₁₁',
                     12: '₁₂', 13: '₁₃', 14: '₁₄', 15: '₁₅', 16: '₁₆', 17: '₁₇', 18: '₁₈', 19: '₁₉', 20: '₂₀', 21: '₂₁',
                     22: '₂₂', 23: '₂₃', 24: '₂₄', 25: '₂₅', 26: '₂₆', 27: '₂₇', 28: '₂₈', 29: '₂₉', 30: '₃₀'}

t = np.linspace(t0, t0 + 3 * T, 1000)
plt.figure(figsize=(12, 6))
plt.plot(t, original_function(t), 'b-', lw=2, label='Исходная функция')
plt.plot(t, trigonometric_sum(t, N, coeffs), 'r--', lw=1.5, label=f'Частичная сумма F{subscript_numbers[N]}(t)')
plt.plot(t, complex_sum(t, N, coeffs), 'g:', lw=1.5, label=f'Частичная сумма G{subscript_numbers[N]}(t)')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.legend()
plt.grid(True)
plt.show()
