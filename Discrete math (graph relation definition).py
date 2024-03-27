print("""Здравствуйте, Павел Андреевич!
Через пробел введите число вершин графа и число рёбер графа: """)
quantity_of_vertices, quantity_of_edges = list(map(int, input().split()))

print("""Для каждого ребра введите через пробел 2 числа:
(номер вершины, ИЗ которой выходит ребро) и (номер вершины, В которую оно идёт): """)

edges = [list(map(int, input(f"Для ребра №{i + 1}: ").split())) for i in range(quantity_of_edges)]
print(edges)


def reflexive():
    count = 0
    for i in range(quantity_of_vertices):  # проверяем, что у каждой вершины есть петля
        if [i + 1, i + 1] in edges:
            count += 1
    if count == quantity_of_vertices:
        return "Рефлексивное"
    elif count == 0:
        return "Антирефлексивное"
    else:
        return "Нерефлексивное"


# ----------------хреновенькое исполнение---------#
#     loops = sum([edge[0] == edge[1] for edge in edges])  # если у одной вершины две петли, то результат будет неверный
#     if loops == quantity_of_vertices:  # количество петель = количеству вершин (у каждой вершины есть петля)
#         return "Рефлексивное"
#     elif loops == 0:                   # ни у одной вершины нет петель
#         return "Антирефлексивное"
#     else:
#         return "Нерефлексивное"


def transitive():
    count = 0  # считаем рёбра, для которых не пополняется транзитивность
    empty_vertices = 0
    for e in edges:  # для каждого ребра
        nice_edges = list(filter(lambda i: i[0] == e[1], edges))  # это список, в котором только рёбра, выходящие из вершины, в которую вошло ребро е
        print("nice_edges", nice_edges)
        if nice_edges == []:  # если ребро ведёт в вершину, из которой потом ничего не растёт, то нафиг его
            empty_vertices += 1
            continue

        for e2 in nice_edges:
            print("нам нужно это ребро, чтоб выполнялась транзитивность: ", [e[0], e2[1]])
            if not ([e[0], e2[1]] in edges):  # if из исходной точки 1‑го ребра нельзя напрямую попасть в конечную точку 2-го ребра
                count += 1
                break  # если хотя бы с одним таким ребром не выполняется транзитивность, то нафиг это вообще, переходим к следующему ребру из исходного списка

    if count + empty_vertices == len(edges):  # если для всех рёбер не выполнилась транзитивность
        return "Антитранзитивное"
    elif count > 0:  # если не выполнилась только для некоторых
        return "Нетранзитивное"
    return "Транзитивное"

def symmetric():
    count = 0
    for edge in edges:
        if (edge[::-1] in edges) and edge[0] != edge[1]:
            count += 1
    if count == len(list(filter(lambda x: x[0] != x[1], edges))):  # для каждого ребра существует такое же, но с противоположным направлением
        return "Симметричное"
    elif count == 0:
        return "Антисимметричное"  # ни для одного ребра не существует такого же, но с противоположным направлением
    return "Несимметричное"


def main():
    r = reflexive()
    t = transitive()
    s = symmetric()
    print(r + "\n" + t + "\n" + s)
    if r == "Антирефлексивное" and s == "Антисимметричное":
        print("\nда ещё и Ассиметричное (сочетание антирефлексивности и антисимметричности)")


main()
