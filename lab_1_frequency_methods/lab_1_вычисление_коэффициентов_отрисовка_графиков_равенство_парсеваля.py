import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

N = 2
a, b = 1, 2
t0, t1, t2 = 3, 4, 5
T = t2 - t0


# Функция для вычисления коэффициентов
def compute_a0(a, b, t0, t1, t2):
    T = t2 - t0
    return (1 / T) * (a * (t1 - t0) + b * (t2 - t1))


def compute_fourier_coefficients(N, a, b, t0, t1, t2):
    T = t2 - t0
    results = {'a': [], 'b': [], 'c': []}

    # a0
    results['a'].append(compute_a0(a, b, t0, t1, t2))

    # an и bn
    for n in range(1, N + 1):
        omega_n = 2 * np.pi * n / T

        # an
        integ_cos = (a * quad(lambda t: np.cos(omega_n * t), t0, t1)[0] +
                     b * quad(lambda t: np.cos(omega_n * t), t1, t2)[0])
        an = (2 / T) * integ_cos
        results['a'].append(an)

        # bn
        integ_sin = (a * quad(lambda t: np.sin(omega_n * t), t0, t1)[0] +
                     b * quad(lambda t: np.sin(omega_n * t), t1, t2)[0])
        bn = (2 / T) * integ_sin
        results['b'].append(bn)

    # cn
    def compute_cn(n):
        omega_n = 2 * np.pi * n / T
        re_integral = (a * quad(lambda t: np.cos(omega_n * t), t0, t1)[0] +
                       b * quad(lambda t: np.cos(omega_n * t), t1, t2)[0])
        im_integral = (a * quad(lambda t: -np.sin(omega_n * t), t0, t1)[0] +
                       b * quad(lambda t: -np.sin(omega_n * t), t1, t2)[0])
        return (re_integral + 1j * im_integral) / T

    for n in range(-N, N + 1):
        results['c'].append((n, compute_cn(n)))

    return results


# Вычисление коэффициентов
coeffs = compute_fourier_coefficients(N, a, b, t0, t1, t2)


# Функции для частичных сумм
def original_function(t):
    """Исходная кусочно-постоянная функция"""
    t_mod = (t - t0) % T + t0  # Приводим к первому периоду
    return np.where(t_mod < t1, a, b)


def trigonometric_sum(t, N, coeffs):
    sum_terms = coeffs['a'][0]  # a0/2

    for n in range(1, N + 1):
        omega_n = 2 * np.pi * n / T
        an = coeffs['a'][n]
        bn = coeffs['b'][n - 1]
        sum_terms += an * np.cos(omega_n * t) + bn * np.sin(omega_n * t)

    return sum_terms


def complex_sum(t, N, coeffs):
    sum_terms = 0

    for n, cn in coeffs['c']:
        if abs(n) > N:
            continue
        omega_n = 2 * np.pi * n / T
        sum_terms += cn * np.exp(1j * omega_n * t)

    return sum_terms.real


# Вывод коэффициентов
print("\nВещественные коэффициенты:")
print(f"a0 = {coeffs['a'][0]:.6f}")  # Вывод a_0
for n in range(1, N + 1):  # Вывод всех a_n и b_n для от 1 до N
    print(f"a{n} = {coeffs['a'][n]:.6f}")
    print(f"b{n} = {coeffs['b'][n - 1]:.6f}")
print("\nКомплексные коэффициенты:")  # Вывод всех c_n для n от -N до N
for n, cn in coeffs['c']:
    print(f"c[{n:2d}] = {cn.real:+.6f} {cn.imag:+.6f}j")


# _______________Равенство Пасеваля______________#

def check_parseval(coeffs, a, b, t0, t1, t2, N):
    T = t2 - t0

    # Левая часть - средняя мощность сигнала
    left_side = (a ** 2 * (t1 - t0) + b ** 2 * (t2 - t1)) / T

    # Правая часть для вещественных коэффициентов
    right_side_real = (coeffs['a'][0] ** 2) + 0.5 * sum([a ** 2 + b ** 2
                                                         for a, b in zip(coeffs['a'][1:], coeffs['b'])])

    # Правая часть для комплексных коэффициентов
    right_side_complex = sum([abs(cn) ** 2 for n, cn in coeffs['c']])

    return left_side, right_side_real, right_side_complex

left, right_real, right_complex = check_parseval(coeffs, a, b, t0, t1, t2, N)

# Вывод результатов
print("\nПроверка равенства Парсеваля:")
print(f"Левая часть (средняя мощность сигнала): {left:.6f}")
print(f"Правая часть (вещественные коэффициенты): {right_real:.6f}")
print(f"Правая часть (комплексные коэффициенты): {right_complex:.6f}")
print(f"Разница (для вещественных): {abs(left - right_real):.6f}")
print(f"Разница (для комплексных): {abs(left - right_complex):.6f}")


# _________________ Построение графиков_____________#

# словарь для красивой подписи нижних индексов на графиках
subscript_numbers = {1: '₁', 2: '₂', 3: '₃', 4: '₄', 5: '₅', 6: '₆', 7: '₇', 8: '₈', 9: '₉', 10: '₁₀', 11: '₁₁',
                     12: '₁₂', 13: '₁₃', 14: '₁₄', 15: '₁₅', 16: '₁₆', 17: '₁₇', 18: '₁₈', 19: '₁₉', 20: '₂₀', 21: '₂₁',
                     22: '₂₂', 23: '₂₃', 24: '₂₄', 25: '₂₅', 26: '₂₆', 27: '₂₇', 28: '₂₈', 29: '₂₉', 30: '₃₀'}


t = np.linspace(t0, t0 + 3 * T + 1, 3000)
original = original_function(t)
trig_approx = trigonometric_sum(t, N, coeffs)
complex_approx = complex_sum(t, N, coeffs)

plt.figure(figsize=(15, 7))
plt.plot(t, original, 'b-', linewidth=3, label='Исходная функция f(t)')
plt.plot(t, trig_approx, 'r--', linewidth=1.5, label=f'Частичная сумма F{subscript_numbers[N]}(t)')
plt.plot(t, complex_approx, 'g:', linewidth=1.5, label=f'Частичная сумма G{subscript_numbers[N]}(t)')
plt.xlabel('t', fontsize=16)
plt.ylabel(f'f(t) и F{subscript_numbers[N]}(t)', fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
for k in range(0, 4):
    plt.axvline(t0 + k * T, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
