import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import tf2zpk, lsim, freqs_zpk
from scipy.fft import fft, ifft, fftshift, ifftshift

plt.rcParams.update({'font.size': 16})


def get_first_order_filter(T):
  return tf2zpk([1], [T, 1])


def filtering_full(a, b, c, d, t1, t2, L, F, T, n=0):
  if not os.path.exists(f'./results_1.1/full/{n}'):
    os.makedirs(f'./results_1.1/{n}')

  # Параметры
  dt_precise = 0.001
  t_precise = np.arange(0, 20, dt_precise)
  N_precise = len(t_precise)
  # Fs_precise = 1/dt_precise
  omega_precise = 2 * np.pi * np.fft.fftfreq(N_precise, dt_precise)
  omega_precise = fftshift(omega_precise)

  # Создание сигналов
  g_precise = np.zeros_like(t_precise)
  g_precise[(t_precise >= t1) & (t_precise <= t2)] = a
  noise_precise = 2 * np.random.rand(N_precise) - 1
  u_precise = g_precise + b * noise_precise

  # Фильтрация во временной области
  filter = get_first_order_filter(T)
  t_filtered, filtered_signal_precise, _ = lsim(filter, u_precise, t_precise)

  # Фурье-анализ
  fourier_g_precise = fftshift(fft(g_precise)) / np.sqrt(N_precise)
  fourier_u_precise = fftshift(fft(u_precise)) / np.sqrt(N_precise)
  W1 = 1 / (T * 1j * omega_precise + 1)
  fourier_filtered_precise = W1 * fourier_u_precise
  freq_filtered_precise = ifft(ifftshift(fourier_filtered_precise)) * np.sqrt(N_precise)

  # Область отображения
  display_range_precise = (t_precise >= t1 - 2) & (t_precise <= t2 + 3)

  # График 1: Сравнение сигналов
  plt.figure(figsize=(12, 8))
  plt.plot(t_precise[display_range_precise], g_precise[display_range_precise], linewidth=3, color='blue')
  plt.plot(t_precise[display_range_precise], u_precise[display_range_precise], linewidth=1.5, color='red', alpha=0.5)
  plt.plot(t_precise[display_range_precise], filtered_signal_precise[display_range_precise], '--', linewidth=1.3,
           color='green')
  plt.title(f'Сравнение g(t), u(t) и u_filt(t) при T={T} и a={a}')
  plt.legend(['g(t)', 'u(t)', 'u_filt(t)'])
  plt.grid()
  plt.xlim([0, t2 + 3])
  plt.xlabel('Время t')
  plt.ylabel('Амплитуда')
  plt.savefig(f'./results_1.1/{n}/1.cравнение_сигналов_{n}.png')
  plt.close()

  # График 2: Сравнение модулей Фурье-образов исходного, зашумлённого и фильтрованного сигналов
  plt.figure(figsize=(12, 8))
  plt.plot(omega_precise, np.abs(fourier_g_precise), linewidth=2, color='blue')
  plt.plot(omega_precise, np.abs(fourier_filtered_precise), linewidth=1.3, color='green')
  plt.plot(omega_precise, np.abs(fourier_u_precise), linewidth=2.8, linestyle='--', color='red', alpha=0.5)
  plt.title(f'Сравнение модулей Фурье-образов g(t), u(t), u_filt(t) при T={T} и a={a}')
  plt.legend(['|g(ω)|', '|u_filt(ω)|', '|u(ω)|'])
  plt.grid()
  plt.xlim([0, 10])
  plt.xlabel('Частота')
  plt.ylabel('Амплитуда')
  plt.savefig(f'./results_1.1/{n}/2.cравнение_модулей_образов{n}.png')
  plt.close()

  # График 3: Сравнение фильтрованного сигнала и обратного преобразования Фурье от W1(iω)·u(ω)
  plt.figure(figsize=(12, 8))
  plt.plot(t_precise[display_range_precise], filtered_signal_precise[display_range_precise], linewidth=2)
  plt.plot(t_precise[display_range_precise], np.real(freq_filtered_precise[display_range_precise]), '--', linewidth=1.5)
  plt.title(f'Сравнение u_filt(t) и ℱ⁻¹[W1(iω)·u(ω)] при T={T} и a={a}')
  plt.legend(['u_filt(t)', '|W1(iω)·u(ω)|'])
  plt.grid()
  plt.xlim([0, t2 + 3])
  plt.xlabel('Время t')
  plt.ylabel('Амплитуда')
  plt.savefig(f'./results_1.1/{n}/3.cравнение_во_временной_области{n}.png')
  plt.close()

  # График 4: Сравнение модуля Фурье-образа фильтрованного сигнала и модуля произведения W1(iω)·u(ω)
  plt.figure(figsize=(12, 8))
  fourier_filtered_time_precise = fftshift(fft(filtered_signal_precise)) / np.sqrt(N_precise)
  plt.plot(omega_precise, np.abs(fourier_filtered_time_precise), linewidth=2)
  plt.plot(omega_precise, np.abs(W1 * fourier_u_precise), linewidth=1.5, linestyle='--')
  plt.title(f'Сравнение |u_filt(ω)| и |W1(iω)·u(ω)| при T={T} и a={a}')
  plt.legend(['|u_filt(ω)|', '|W1(iω)·u(ω)|'])
  plt.grid()
  plt.xlim([0, 10])
  plt.xlabel('Частота')
  plt.ylabel('Амплитуда')
  plt.savefig(f'./results_1.1/{n}/4.cравнение_в_частотной_области{n}.png')
  plt.close()

  # График 5: АЧХ
  plt.figure(figsize=(12, 8))
  plt.plot(omega_precise[omega_precise >= 0], np.abs(W1[omega_precise >= 0]), 'b', linewidth=2)
  plt.title(f'АЧХ фильтра 1-го порядка (T={T})', fontsize=16)
  plt.xlabel('Частота (рад/с)', fontsize=14)
  plt.ylabel('Коэффициент передачи', fontsize=14)
  plt.grid()

  # Добавляем линию на уровне 1/sqrt(2) для определения частоты среза
  plt.axhline(y=1 / np.sqrt(2), color='r', linestyle='--')
  plt.text(0, 1 / np.sqrt(2) + 0.05, r'$\frac{1}{\sqrt{2}}$', fontsize=14)

  # Определяем частоту среза (частота, при которой АЧХ = 1/sqrt(2))
  w_cut = 1 / T  # Теоретическая частота среза для фильтра 1-го порядка
  plt.axvline(x=w_cut, color='g', linestyle=':')
  plt.text(w_cut + 0.5, 0.1, f'ω_c={w_cut:.2f} рад/с', fontsize=12)

  plt.xlim([0, 2 * w_cut])  # Ограничиваем диапазон частот
  plt.savefig(f'./results_1.1/{n}/5.АЧХ{n}.png')
  plt.close()


# TASK 1.1
filtering_full(a=0.5, b=1.0, c=0, d=0, t1=1, t2=4, L=10, F=10, T=0.01, n=1)
filtering_full(a=2, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, T=0.01, n=2)
filtering_full(a=5, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, T=0.01, n=3)
filtering_full(a=10, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, T=0.01, n=4)

filtering_full(a=5, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, T=1, n=5)
filtering_full(a=5, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, T=0.7, n=6)
filtering_full(a=5, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, T=0.3, n=7)
filtering_full(a=5, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, T=0.1, n=8)
