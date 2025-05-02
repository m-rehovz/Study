import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

N = 2
T = 2 * np.pi
t0 = -np.pi  # Начало периода (симметричный интервал)

def original_function(t):
    return np.sin(-t) / (2 + np.cos(-t)) + np.sin(-t) ** 13

def compute_fourier_coefficients(N):
    results = {'a': [], 'b': [], 'c': []}

    a0 = (1 / T) * quad(lambda t: original_function(t), t0, t0 + T,
                        epsabs=1e-12, epsrel=1e-12)[0]
    results['a'].append(a0)

    for n in range(1, N + 1):
        an = (2 / T) * quad(lambda t: original_function(t) * np.cos(n * t), t0, t0 + T,
                            epsabs=1e-12)[0]
        results['a'].append(an)

        bn = (2 / T) * quad(lambda t: original_function(t) * np.sin(n * t), t0, t0 + T,
                            epsabs=1e-12)[0]
        results['b'].append(bn)


        cn = (an - 1j * bn) / 2
        c_neg_n = (an + 1j * bn) / 2
        results['c'].extend([(n, cn), (-n, c_neg_n)])

    results['c'].append((0, a0 / 2))
    results['c'].sort(key=lambda x: x[0])

    return results


def trigonometric_sum(t, N, coeffs):
    sum_terms = coeffs['a'][0] / 2
    for n in range(1, N + 1):
        sum_terms += coeffs['a'][n] * np.cos(n * t) + coeffs['b'][n - 1] * np.sin(n * t)
    return sum_terms


def complex_sum(t, N, coeffs):
    sum_terms = 0
    for n, cn in coeffs['c']:
        if abs(n) > N:
            continue
        sum_terms += cn * np.exp(1j * n * t)
    return sum_terms.real


coeffs = compute_fourier_coefficients(N)

print("=" * 50)
print("Вещественные коэффициенты:")
print(f"a0 = {coeffs['a'][0]:.10f} (должен быть близок к 0)")
for n in range(1, N + 1):
    print(f"a{n} = {coeffs['a'][n]:.10f}, b{n} = {coeffs['b'][n - 1]:.10f}")

print("\nКомплексные коэффициенты:")
for n, cn in coeffs['c']:
    print(f"c[{n:2d}] = {cn.real:+.10f} {cn.imag:+.10f}j")

# Проверка равенства Парсеваля
print("\n" + "=" * 50)
power_original = (1 / T) * quad(lambda t: original_function(t) ** 2, t0, t0 + T, epsabs=1e-12)[0]
power_real = (coeffs['a'][0] ** 2) / 2 + 0.5 * sum(a ** 2 + b ** 2 for a, b in zip(coeffs['a'][1:], coeffs['b']))
power_complex = sum(abs(cn) ** 2 for n, cn in coeffs['c'])

print("Проверка равенства Парсеваля:")
print(f"Оригинальная мощность: {power_original:.12f}")
print(f"По вещественным коэф: {power_real:.12f}")
print(f"По комплексным коэф:  {power_complex:.12f}")
print(f"\nРазница (вещ.): {abs(power_original - power_real):.12f}")
print(f"Разница (комп.): {abs(power_original - power_complex):.12f}")

# Построение графиков
subscript_numbers = {1: '₁', 2: '₂', 3: '₃', 4: '₄', 5: '₅', 6: '₆', 7: '₇', 8: '₈', 9: '₉', 10: '₁₀', 11: '₁₁',
                     12: '₁₂', 13: '₁₃', 14: '₁₄', 15: '₁₅', 16: '₁₆', 17: '₁₇', 18: '₁₈', 19: '₁₉', 20: '₂₀', 21: '₂₁',
                     22: '₂₂', 23: '₂₃', 24: '₂₄', 25: '₂₅', 26: '₂₆', 27: '₂₇', 28: '₂₈', 29: '₂₉', 30: '₃₀'}


t = np.linspace(-2*T, t0 + 3 * T, 1000)
plt.figure(figsize=(12, 6))
plt.plot(t, original_function(t), 'b-', lw=2, label='Исходная функция')
plt.plot(t, trigonometric_sum(t, N, coeffs), 'r--', lw=1.5, label=f'Частичная сумма F{subscript_numbers[N]}(t)')
plt.plot(t, complex_sum(t, N, coeffs), 'g:', lw=1.5, label=f'Частичная сумма G{subscript_numbers[N]}(t)')
plt.xlabel('t');
plt.ylabel('f(t)')
plt.legend();
plt.grid(True)
plt.show()