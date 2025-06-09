import numpy as np 
import matplotlib.pyplot as plt 
import os
import time

def rect_func(t):
    f = (np.abs(t) <= 1/2)

    return f

def num_fourier(t, v, func):
    exponentials = np.exp(-2j * np.pi * v[:, None] * t[None, :])
    image = np.trapezoid(func[None, :] * exponentials, t, axis=1)
    
    return image

def num_inverse_fourier(v, t, func):
    exponentials = np.exp(2j * np.pi * t[:, None] * v[None, :])
    inverse = np.trapezoid(func[None, :] * exponentials, v, axis=1)
    
    return inverse

def fft_fourier(dt, func):
    N = len(func)
    v = np.fft.fftshift(np.fft.fftfreq(N, dt))
    image = np.fft.fftshift(np.fft.fft(func)) * dt 
    return v, image

def ifft_fourier(dt, func):
    N = len(func)
    inverse = np.fft.ifft(np.fft.ifftshift(func)) / dt
    return inverse

def cont_fourier(dt, func):
    N = len(func)
    v = np.fft.fftshift(np.fft.fftfreq(N, dt))
    image = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(func))) * dt 
    return v, image


def icont_fourier(dt, func):
    inverse = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(func))) / dt 
    return inverse

def original_func_image():
    T, dt = 10, 0.001
    V, dv = 20, 0.001

    t = np.arange(-T/2, T/2, dt)
    v = np.arange(-V/2, V/2, dv)

    f_orig = rect_func(t)
    f_orig_image = np.sinc(v)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, f_orig, linewidth=2, color='b')
    plt.grid()
    plt.title('Оригинальная функция')

    plt.subplot(2, 1, 2)
    plt.plot(v, f_orig_image, linewidth=2, color='b')
    plt.grid()
    plt.title('Аналитический Фурье-образ')

    plt.tight_layout()
    plt.savefig('./plots/task1/original.png')
    plt.close()


def issled_trapz(T, dt, V, dv, n):
    t_orig = np.arange(-30, 30, 0.001)
    v_orig = np.arange(-30, 30, 0.001)
    f_orig = rect_func(t_orig)
    f_orig_image = np.sinc(v_orig)
    
    t = np.arange(-T/2, T/2, dt)
    v = np.arange(-V/2, V/2, dv)
    f = rect_func(t)
    
    timer_trapz = time.time()
    f_image = num_fourier(t, v, f)
    timer_trapz = time.time() - timer_trapz

    timer_inverse = time.time()
    f_restored = num_inverse_fourier(v, t, f_image)
    timer_inverse = time.time() - timer_inverse

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_orig, f_orig, linewidth=2, label=f'Оригинальная функция', color='b')
    plt.plot(t, f_restored, linewidth=2, linestyle='--', label='Численное интегрирование', color='salmon')
    plt.title("Функции во временной области")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$f(t)$")
    plt.xlim([-5, 5])
    plt.legend(loc='upper right')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(v_orig, f_orig_image, linewidth=2, label=f'Оригинальный образ', color='b')
    plt.plot(v, f_image, linewidth=2, linestyle='--', label='Численное интегрирование', color='salmon')
    plt.title("Фурье-образы")
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$\hat{f}(\nu)$")
    plt.xlim([-20, 20])
    plt.legend(loc='upper right')
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'./plots/task1/trapz/{n}.png')
    plt.close()

    print(f'Время поиска Фурье-образа {n}:  {timer_trapz:.3f}')
    print(f'Время поиска восстановленной функции {n}: {timer_inverse:.3f}')

def issled_dft(T, dt, n):
    t_orig = np.arange(-30, 30, 0.001)
    v_orig = np.arange(-30, 30, 0.001)
    f_orig = rect_func(t_orig)
    f_orig_image = np.sinc(v_orig)
    
    t = np.arange(-T/2, T/2, dt)
    f = rect_func(t)
    
    timer_fft = time.time()
    v, f_image = fft_fourier(dt, f)
    timer_fft = time.time() - timer_fft

    timer_ifft = time.time()
    f_restored = ifft_fourier(dt, f_image)
    timer_ifft = time.time() - timer_ifft

    V, dv = round(max(v), 1), round(v[1] - v[0], 6)
    print(f'для n = {n} V = {V}, dv = {dv}')

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_orig, f_orig, linewidth=2, label=f'Оригинальная функция', color='b')
    plt.plot(t, f_restored, linewidth=2, linestyle='--', label='Дискретное преобразование', color='salmon')
    plt.title("Функции во временной области")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$f(t)$")
    plt.xlim([-5, 5])
    plt.legend(loc='upper right')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(v_orig, f_orig_image, linewidth=2, label=f'Оригинальный образ', color='b')
    plt.plot(v, f_image, linewidth=2, linestyle='--', label='Дискретное преобразование', color='salmon')
    plt.title("Фурье-образы")
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$\hat{f}(\nu)$")
    plt.xlim([-20, 20])
    plt.legend(loc='upper right')
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'./plots/task1/dft/{n}.png')
    plt.close()

    print(f'Время поиска Фурье-образа {n}:  {timer_fft:.8f}')
    print(f'Время поиска восстановленной функции {n}: {timer_ifft:.8f}')


def issled_cont(T, dt, n):
    t_orig = np.arange(-30, 30, 0.001)
    v_orig = np.arange(-30, 30, 0.001)
    f_orig = rect_func(t_orig)
    f_orig_image = np.sinc(v_orig)
    
    t = np.arange(-T/2, T/2, dt)
    f = rect_func(t)
    
    timer_cont = time.time()
    v, f_image = cont_fourier(dt, f)
    timer_cont = time.time() - timer_cont

    timer_icont = time.time()
    f_restored = icont_fourier(dt, f_image)
    timer_icont = time.time() - timer_icont

    V, dv = round(max(v), 1), round(v[1] - v[0], 6)
    print(f'для n = {n} V = {V}, dv = {dv}')

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_orig, f_orig, linewidth=2, label=f'Оригинальная функция', color='b')
    plt.plot(t, f_restored, linewidth=2, linestyle='--', label='Непрерывное преобразование', color='salmon')
    plt.title("Функции во временной области")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$f(t)$")
    plt.xlim([-5, 5])
    plt.legend(loc='upper right')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(v_orig, f_orig_image, linewidth=2, label=f'Оригинальный образ', color='b')
    plt.plot(v, f_image, linewidth=2, linestyle='--', label='Непрерывное преобразование', color='salmon')
    plt.title("Фурье-образы")
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$\hat{f}(\nu)$")
    plt.xlim([-20, 20])
    plt.legend(loc='upper right')
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'./plots/task1/cont/{n}.png')
    plt.close()

    print(f'Время поиска Фурье-образа {n}:  {timer_cont:.8f}')
    print(f'Время поиска восстановленной функции {n}: {timer_icont:.8f}')


# original_func_image()

# T, dt, V, dv = 1, 0.01, 20, 0.01
# issled_trapz(T, dt, V, dv, 1)
# T, dt, V, dv = 5, 0.01, 20, 0.01
# issled_trapz(T, dt, V, dv, 2)
# T, dt, V, dv = 10, 0.01, 20, 0.01
# issled_trapz(T, dt, V, dv, 3)

# T, dt, V, dv = 10, 1, 20, 0.01
# issled_trapz(T, dt, V, dv, 4)
# T, dt, V, dv = 10, 0.1, 20, 0.01
# issled_trapz(T, dt, V, dv, 5)
# T, dt, V, dv = 10, 0.01, 20, 0.01
# issled_trapz(T, dt, V, dv, 6)

# T, dt, V, dv = 10, 0.01, 1, 0.01
# issled_trapz(T, dt, V, dv, 7)
# T, dt, V, dv = 10, 0.01, 5, 0.01
# issled_trapz(T, dt, V, dv, 8)
# T, dt, V, dv = 10, 0.01, 20, 0.01
# issled_trapz(T, dt, V, dv, 9)

# T, dt, V, dv = 10, 0.01, 20, 1
# issled_trapz(T, dt, V, dv, 10)
# T, dt, V, dv = 10, 0.01, 20, 0.1
# issled_trapz(T, dt, V, dv, 11)
# T, dt, V, dv = 10, 0.01, 20, 0.01
# issled_trapz(T, dt, V, dv, 12)

# # для n = 1 V = 49.0, dv = 1.0
# T, dt = 1, 0.01
# issled_dft(T, dt, 1)
# # для n = 2 V = 49.8, dv = 0.2
# T, dt = 5, 0.01
# issled_dft(T, dt, 2)
# # для n = 3 V = 49.9, dv = 0.1
# T, dt = 10, 0.01
# issled_dft(T, dt, 3)

# # для n = 4 V = 0.4, dv = 0.1
# T, dt = 10, 1
# issled_dft(T, dt, 4)
# # для n = 5 V = 4.9, dv = 0.1
# T, dt = 10, 0.1
# issled_dft(T, dt, 5)
# # для n = 6 V = 49.9, dv = 0.1
# T, dt = 10, 0.01
# issled_dft(T, dt, 6)

# # для n = 1 V = 49.0, dv = 1.0
# T, dt = 1, 0.01
# issled_cont(T, dt, 1)
# # для n = 2 V = 49.8, dv = 0.2
# T, dt = 5, 0.01
# issled_cont(T, dt, 2)
# # для n = 3 V = 49.9, dv = 0.1
# T, dt = 10, 0.01
# issled_cont(T, dt, 3)

# # для n = 4 V = 0.4, dv = 0.1
# T, dt = 10, 1
# issled_cont(T, dt, 4)
# # для n = 5 V = 4.9, dv = 0.1
# T, dt = 10, 0.1
# issled_cont(T, dt, 5)
# # для n = 6 V = 49.9, dv = 0.1
# T, dt = 10, 0.01
# issled_cont(T, dt, 6)


T, dt = 100, 0.001
a1, w1, p1, a2, w2, p2 = 4, 5, 3, 3, 7, 1
b = 5

def sinc_interp(t, t_samples, f_samples, B):
    result = np.zeros_like(t)
    for n in range(len(t_samples)):
        result += f_samples[n] * np.sinc(2 * B * (t - t_samples[n]))
    return result

def issled_first(dt_sampled, T_sampled, n):
    t = np.arange(-T/2, T/2, dt)
    B = 1 / (2 * dt_sampled)
    print(f'Найденное B = {B} для n = {n}')
    t_sampled = np.arange(-T_sampled/2, T_sampled/2, dt_sampled)

    y = a1 * np.sin(w1 * t + p1) + a2 * np.sin(w2 * t + p2)
    y_sampled = a1 * np.sin(w1 * t_sampled + p1) + a2 * np.sin(w2 * t_sampled + p2)

    y_recovery = sinc_interp(t, t_sampled, y_sampled, B)

    v_cont, image_cont = cont_fourier(dt, y)
    v_sampled, image_sampled = cont_fourier(dt_sampled, y_sampled)
    v_rec, image_rec = cont_fourier(dt, y_recovery)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, y, linewidth=2, label='Непрерывная', color='darkgrey')
    plt.vlines(t_sampled, 0, y_sampled, colors='b', linewidth=2, label='Сэмплы')
    plt.plot(t, y_recovery, linewidth=2, linestyle='--', label=f'Восстановленная', color='salmon')
    plt.title("Функции во временной области")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$f(t)$")
    plt.xlim([-10, 10])
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(v_cont, image_cont, linewidth=2, label='Непрерывная', color='darkgrey')
    plt.plot(v_sampled, image_sampled, linewidth=2, linestyle=':', label='Сэмплы', color='b')
    plt.plot(v_rec, image_rec, linewidth=2, linestyle='--', label='Восстановленная', color='salmon')
    plt.axvspan(-B, B, color='black', alpha=0.1)
    plt.axvline(x=-B, color='black', linestyle='--')
    plt.axvline(x=B, color='black', linestyle='--', label='Границы частот')
    plt.title("Фурье-образы")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$f(t)$")
    plt.xlim([-10, 10])
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'./plots/task2/first/{n}.png')
    plt.close()

def issled_second(dt_sampled, T_sampled, n):
    t = np.arange(-T/2, T/2, dt)
    B = 1 / (2 * dt_sampled)
    print(f'Найденное B = {B} для n = {n}')
    t_sampled = np.arange(-T_sampled/2, T_sampled/2, dt_sampled)

    y = np.sinc(b * t)
    y_sampled = np.sinc(b * t_sampled)

    y_recovery = sinc_interp(t, t_sampled, y_sampled, B)

    v_cont, image_cont = cont_fourier(dt, y)
    v_sampled, image_sampled = cont_fourier(dt_sampled, y_sampled)
    v_rec, image_rec = cont_fourier(dt, y_recovery)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, y, linewidth=2, label='Непрерывная', color='darkgrey')
    plt.vlines(t_sampled, 0, y_sampled, colors='b', linewidth=2, label='Сэмплы')
    plt.plot(t, y_recovery, linewidth=2, linestyle='--', label=f'Восстановленная', color='salmon')
    plt.title("Функции во временной области")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$f(t)$")
    plt.xlim([-10, 10])
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(v_cont, image_cont, linewidth=2, label='Непрерывная', color='darkgrey')
    plt.plot(v_sampled, image_sampled, linewidth=2, linestyle=':', label='Сэмплы', color='b')
    plt.plot(v_rec, image_rec, linewidth=2, linestyle='--', label='Восстановленная', color='salmon')
    plt.axvspan(-B, B, color='black', alpha=0.1)
    plt.axvline(x=-B, color='black', linestyle='--')
    plt.axvline(x=B, color='black', linestyle='--', label='Границы частот')
    plt.title("Фурье-образы")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$f(t)$")
    plt.xlim([-10, 10])
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'./plots/task2/second/{n}.png')
    plt.close()

# Найденное B = 0.5 для n = 1
dt_sampled, T_sampled = 1, 100
issled_first(dt_sampled, T_sampled, 1)
# Найденное B = 1.0 для n = 2
dt_sampled, T_sampled = 0.5, 100
issled_first(dt_sampled, T_sampled, 2)
# Найденное B = 2.5 для n = 3
dt_sampled, T_sampled = 0.2, 100
issled_first(dt_sampled, T_sampled, 3)
# Найденное B = 5.0 для n = 4
dt_sampled, T_sampled = 0.2, 5
issled_first(dt_sampled, T_sampled, 4)

dt_sampled, T_sampled = 1, 100
issled_second(dt_sampled, T_sampled, 1)
dt_sampled, T_sampled = 0.5, 100
issled_second(dt_sampled, T_sampled, 2)
dt_sampled, T_sampled = 0.2, 100
issled_second(dt_sampled, T_sampled, 3)
dt_sampled, T_sampled = 0.2, 5
issled_second(dt_sampled, T_sampled, 4)




