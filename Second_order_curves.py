import math as m, numpy as np

def povorot(A, B, C, D, E, F, axaxa=False):
    if A == C: alfa = m.pi / 4
    else: alfa = m.atan(B / (C - A)) / 2  # tg(2alfa) = B/(A-C)
    if axaxa == True: alfa = m.pi / 2
    A_1 = A * m.cos(alfa) ** 2 - B * m.cos(alfa) * m.sin(alfa) + C * m.sin(alfa) ** 2    # расчёт коэффициентов общего уравнения кривой
    C_1 = A * m.sin(alfa) ** 2 + B * m.sin(alfa) * m.cos(alfa) + C * m.cos(alfa) ** 2
    B_1 = 2 * A * m.sin(alfa) * m.cos(alfa) + B * m.cos(alfa) ** 2 - B * m.sin(alfa) ** 2 - 2 * C * m.sin(alfa) * m.cos(alfa)
    D_1 = D * m.cos(alfa) - E * m.sin(alfa)
    E_1 = D * m.sin(alfa) + E * m.cos(alfa)
    F_1 = F - D_1 ** 2 / (4 * A_1) - E_1 ** 2 / (4 * C_1)
    if B_1 < 10 ** (-10): B_1 = 0
    return [A_1, B_1, C_1, D_1, E_1, F_1]


def curve_classification( A, B, C, D, E, F):
    def straight_line_classification():
        k = np.linalg.det(np.array([[A, D / 2], [D / 2, F]])) + np.linalg.det(np.array([[C, E / 2], [E / 2, F]]))
        if k < 0: print('Кривая первого порядка: пара параллельных прямых')
        elif k > 0: print('Alert! У вас пустое множество! Alert!')
        else: print('Кривая первого порядка: пара совпадающих прямых (нет блин тройка параллельных кривых) (одна прямая получается)')

    def ellipse(A, B, C, D, E, F):
        A, B, C, D, E, F = povorot(A, B, C, D, E, F)
        a_qv, b_qv = abs(1 / A), abs(1 / C)
        c = m.sqrt(abs(a_qv - b_qv))
        while a_qv < b_qv:
            A, B, C, D, E, F = povorot(A, B, C, D, E, F, axaxa=True)
            a_qv, b_qv = abs(1 / A), abs(1 / C)
            c = m.sqrt(a_qv - b_qv)
        e = round(c/m.sqrt(a_qv), 3)
        print(f'Эксцентриситет равен {e}!')
        if e == 0: print('Это окружность')

    def hyperbola(A, B, C, D, E, F):
        if B != 0: A, B, C, D, E, F = povorot(A, B, C, D, E, F)
        a_qv, b_qv = abs(1 / A), abs(1 / C)
        c = m.sqrt(a_qv + b_qv)
        print(f'Эксцентриситет равен {round(c/m.sqrt(a_qv), 90)}')

    def analysis(delta):
        P_det = np.linalg.det(P)
        if P_det == delta and delta == 0: straight_line_classification()
        if delta > 0 and P_det != 0 and tau * P_det < 0:
            print('Эллипс')
            ellipse(A, B, C, D, E, F)
        elif delta > 0 and P_det != 0 and tau * P_det > 0: print('Alert! У вас пустое множество! Alert!')
        elif delta > 0 and P_det == 0: print('Точка')
        elif delta < 0 and P_det != 0:
            print('Гипербола')
            hyperbola(A, B, C, D, E, F)
        elif delta < 0 and P_det == 0: print('Кривая первого порядка: пара пересекающихся прямых')
        elif delta == 0 and P_det != 0: print('Парабола, эксцентриситет равен 1')

    P, delta, tau = np.array([[A, B / 2, D / 2, ], [B / 2, C, E / 2], [D / 2, E / 2, F]]), np.linalg.det(np.array([[A, B / 2], [B / 2, C]])), A + C
    discr = tau ** 2 - 4 * delta
    if discr >= 0: analysis(delta)
    elif discr < 0: print('Тут дискриминант отрицательный')

def main():
    def input_data():
        entered_list = input('Введите 6 чисел через запятую: ').split(',')
        while True:
            try:
                entered_list_1 = list(map(lambda x: float(x), entered_list))
                if len(entered_list_1) != 6: raise ZeroDivisionError
                break
            except:
                entered_list = input('Введите 6 чисел ещё раз, вы дурачок: ').split(',')
        return entered_list_1

    print('Приветствую вас!')
    data = input_data()
    if set(data) == {0}: print('Поздравляем! У вас вся плоскость!')
    else:
        A, B, C, D, E, F = data
        curve_classification(A, B, C, D, E, F)

main()