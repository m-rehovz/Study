import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import lsim, tf2zpk, zpk2tf  # Добавляем zpk2tf для преобразования
from scipy.fft import fft, ifft, fftshift, ifftshift

omega_0 = 10

plt.rcParams.update({'font.size': 16})


# Функция для фильтра второго порядка
def get_second_order_filter(a1, a2, b1, b2):
    # Числитель и знаменатель для W_2(p)
    num = [1, a1, a2]  # [p^2, a1*p, a2]
    den = [1, b1, b2]  # [p^2, b1*p, b2]
    zeros, poles, k = tf2zpk(num, den)
    return zeros, poles, k

def filtering_full(a, c, d, t1, t2, a1, a2, b1, b2, n=0):
    if not os.path.exists(f'./results_1.2/{n}'):
        os.makedirs(f'./results_1.2/{n}')

    # Параметры
    dt_precise = 0.0001
    t_precise = np.arange(0, 20, dt_precise)
    N_precise = len(t_precise)
    omega_precise = 2 * np.pi * np.fft.fftfreq(N_precise, dt_precise)
    omega_precise = fftshift(omega_precise)

    # Создание сигналов
    g_precise = np.zeros_like(t_precise)
    g_precise[(t_precise >= t1) & (t_precise <= t2)] = a
    # Синусоидальная помеха
    u_precise = g_precise + c * np.sin(d * t_precise)

    # Фильтрация во временной области с фильтром второго порядка
    zeros, poles, k = get_second_order_filter(a1, a2, b1, b2)
    t_filtered, filtered_signal_precise, _ = lsim((zeros, poles, k), u_precise, t_precise)

    # Фурье-анализ
    fourier_g_precise = fftshift(fft(g_precise)) / np.sqrt(N_precise)
    fourier_u_precise = fftshift(fft(u_precise)) / np.sqrt(N_precise)

    # Передаточная функция W2 в частотной области
    s = 1j * omega_precise
    W2 = (s**2 + a1 * s + a2) / (s**2 + b1 * s + b2)
    fourier_filtered_precise = W2 * fourier_u_precise
    freq_filtered_precise = ifft(ifftshift(fourier_filtered_precise)) * np.sqrt(N_precise)

    # Область отображения
    display_range_precise = (t_precise >= t1 - 2) & (t_precise <= t2 + 3)

    # График 1: Сравнение сигналов
    plt.figure(figsize=(12, 8))
    plt.plot(t_precise[display_range_precise], g_precise[display_range_precise], linewidth=3, color='blue')
    plt.plot(t_precise[display_range_precise], u_precise[display_range_precise], linewidth=1.5, color='red', alpha=0.5)
    plt.plot(t_precise[display_range_precise], filtered_signal_precise[display_range_precise], '--', linewidth=1.3, color='green')
    plt.title(f'Сравнение g(t), u(t) и u_filt(t) при a1={a1}, a2={a2}, b1={b1}, b2={b2}, c={c}, d={d}')
    plt.legend(['g(t)', 'u(t)', 'u_filt(t)'])
    plt.grid()
    plt.xlim([0, t2 + 3])
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.savefig(f'./results_1.2/{n}/1_comparison_signals_{n}.png')
    plt.close()


    # График 2: Сравнение модулей Фурье-образов
    plt.figure(figsize=(12, 8))
    plt.plot(omega_precise, np.abs(fourier_g_precise), linewidth=2, color='blue')
    plt.plot(omega_precise, np.abs(fourier_u_precise), linewidth=1.5, linestyle='--', color='red', alpha=0.5)
    plt.plot(omega_precise, np.abs(fourier_filtered_precise), linewidth=1.3, color='green')
    plt.title(f'Сравнение |g(ω)|, |u(ω)|, |u_filt(ω)| при a1={a1}, a2={a2}, b1={b1}, b2={b2}')
    plt.legend(['|g(ω)|', '|u(ω)|', '|u_filt(ω)|'])
    plt.grid()
    plt.xlim([0, 2 * max(d, 10)])
    plt.xlabel('Частота (рад/с)')
    plt.ylabel('Амплитуда')
    plt.savefig(f'./results_1.2/{n}/2_comparison_spectra_{n}.png')
    plt.close()


    # График 3: Сравнение фильтрованного сигнала и обратного преобразования
    plt.figure(figsize=(12, 8))
    plt.plot(t_precise[display_range_precise], filtered_signal_precise[display_range_precise], linewidth=2, color='blue')
    plt.plot(t_precise[display_range_precise], np.real(freq_filtered_precise[display_range_precise]), '--', linewidth=1.5, color='green')
    plt.title(f'Сравнение u_filt(t) и ℱ⁻¹[W2(iω)·u(ω)] при a1={a1}, a2={a2}, b1={b1}, b2={b2}')
    plt.legend(['u_filt(t)', 'ℱ⁻¹[W2(iω)·u(ω)]'])
    plt.grid()
    plt.xlim([0, t2 + 3])
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.savefig(f'./results_1.2/{n}/3_comparison_time_{n}.png')
    plt.close()


    # График 4: Сравнение модулей в частотной области
    plt.figure(figsize=(12, 8))
    fourier_filtered_time_precise = fftshift(fft(filtered_signal_precise)) / np.sqrt(N_precise)
    plt.plot(omega_precise, np.abs(fourier_filtered_time_precise), linewidth=2, color='blue')
    plt.plot(omega_precise, np.abs(W2 * fourier_u_precise), linewidth=1.5, linestyle='--', color='green')
    plt.title(f'Сравнение |u_filt(ω)| и |W2(iω)·u(ω)| при a1={a1}, a2={a2}, b1={b1}, b2={b2}')
    plt.legend(['|u_filt(ω)|', '|W2(iω)·u(ω)|'])
    plt.grid()
    plt.xlim([0, 2 * max(d, 10)])
    plt.xlabel('Частота (рад/с)')
    plt.ylabel('Амплитуда')
    plt.savefig(f'./results_1.2/{n}/4_comparison_freq_{n}.png')
    plt.close()


    # График 5: АЧХ
    plt.figure(figsize=(12, 8))
    plt.plot(omega_precise[omega_precise >= 0], np.abs(W2[omega_precise >= 0]), 'b', linewidth=2)
    plt.title(f'АЧХ фильтра 2-го порядка при a1={a1}, a2={a2}, b1={b1}, b2={b2}', fontsize=16)
    plt.xlabel('Частота (рад/с)', fontsize=14)
    plt.ylabel('Коэффициент передачи', fontsize=14)
    plt.grid()

    # Определяем частоту ω0 (нулевая точка АЧХ)
    w0_approx = np.sqrt(a2)
    plt.axhline(y=0, color='r', linestyle='--', label='АЧХ = 0')
    plt.axvline(x=w0_approx, color='g', linestyle=':', label=f'ω0 ≈ {w0_approx:.2f} рад/с')
    plt.legend()
    plt.xlim([0, 2 * max(d, w0_approx)])
    plt.savefig(f'./results_1.2/{n}/5_ACH_{n}.png')
    plt.close()


# TASK 1.2
a1, a2, b1, b2 = 0, 100, 2*0.1*omega_0, 100  # ω0 ≈ 10 рад/с

# Базовый случай
filtering_full(a=5, c=1, d=10, t1=1, t2=4, a1=a1, a2=a2, b1=2 * 0.5 * omega_0, b2=b2, n=1)

# Влияние частоты d
filtering_full(a=5, c=1, d=8, t1=1, t2=4, a1=a1, a2=a2, b1=2 * 0.5 * omega_0, b2=b2, n=2)
filtering_full(a=5, c=1, d=15, t1=1, t2=4, a1=a1, a2=a2, b1=2 * 0.5 * omega_0, b2=b2, n=3)

# Влияние ζ (b1)
filtering_full(a=5, c=1, d=10, t1=1, t2=4, a1=a1, a2=a2, b1=2 * 1.5 * omega_0, b2=b2, n=4)
filtering_full(a=5, c=1, d=10, t1=1, t2=4, a1=a1, a2=a2, b1=2 * 0.05 * omega_0, b2=b2, n=5)
