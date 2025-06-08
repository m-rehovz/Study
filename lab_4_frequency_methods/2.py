import pandas as pd

from scipy import signal
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 16})
colors = ['blue', 'green', 'red', 'purple', 'orange']

data = pd.read_csv('SBER.csv', sep=';')
prices = data['<CLOSE>'].values
dates = pd.to_datetime(data['<DATE>'], format='%y%m%d')

# Параметры
T_values = [1, 5, 21, 63, 252]  # Постоянные времени в днях
dt = 1

# Функция для создания фильтра первого порядка
def create_first_order_filter(T, dt):
    system = signal.TransferFunction([1], [T, 1])
    discrete_system = system.to_discrete(dt=dt, method='bilinear')
    return discrete_system.num, discrete_system.den

# Фильтрация и сохранение результатов
filtered_signals = {}
for T in T_values:
    b, a = create_first_order_filter(T, dt)
    filtered_signal = signal.filtfilt(b, a, prices)
    filtered_signals[T] = filtered_signal

output_dir = './results_2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Построение пяти отдельных графиков (исходные данные + фильтрованный сигнал для каждого T)
for i, T in enumerate(T_values):
    plt.figure(figsize=(12, 8))
    plt.plot(dates, prices, linewidth=2, color='black', alpha=0.5, label='Исходные данные')
    plt.plot(dates, filtered_signals[T], linewidth=1.5, linestyle='--', color=colors[i],
             label=f'Фильтр, T={T} дней')
    plt.title(f'Сравнение исходных и фильтрованных данных (T={T} дней)')
    plt.xlabel('Дата')
    plt.ylabel('Цена закрытия (RUB)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sber_filtered_T{T}.png')
    plt.close()

# Все фильтрованные сигналы + исходные данные
plt.figure(figsize=(12, 8))
plt.plot(dates, prices, label='Исходные данные', color='black', alpha=0.5, linewidth=2)
for i, T in enumerate(T_values):
    plt.plot(dates, filtered_signals[T], label=f'T={T} дней', color=colors[i], linewidth=1.5, linestyle='--')
plt.title('Сглаживание биржевых данных SBER')
plt.xlabel('Дата')
plt.ylabel('Цена закрытия (RUB)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/sber_filtered_all.png')
plt.show()
