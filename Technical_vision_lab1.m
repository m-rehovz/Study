clc
clear all

I = imread("cat2.png");
[numRows, numCols, Layers] = size(I);

%% ГИСТОГРАММЫ

figure('Name','Гистограмма оригинального изображения','NumberTitle','off');

subplot(2,3,4);
imhist(I(:, :, 1));  % красный канал
title("Red");

subplot(2,3,5);
imhist(I(:, :, 2));  % зелёный канал
title("Green");

subplot(2,3,6);
imhist(I(:, :, 3));  % синий канал
title("Blue");

subplot(2,3,2);
imshow(I);
%% АРИФМЕТИЧЕСКИЕ ОПЕРАЦИИ

Inew = I;
for k = 1:1: Layers   %для каждого цветового канала
    for i = 1:1: numRows   % для каждой строки
        for j = 1:1: numCols   % для каждого столбца
            Inew(i,j, k) = Inew(i,j,k) - 50;
        end
    end
end

printComparison(I, Inew, 'Арифметические операции (-50 по яркости)')

%% РАСТЯЖЕНИЕ ДИНАМИЧЕСКОГО ДИАПАЗОНА

Inew1 = I;
alfa = 4;   % даёт красочную картинку
for k = 1:1: Layers
    Imin = min(min(I(: , : , k)));
    Imax = max(max(I(: , : , k)));
    for i = 1:1: numRows
        for j = 1:1: numCols
            Inew1(i ,j , k) = (double((Inew1(i, j, k) - Imin)) / double((Imax - Imin )))^alfa * 256;   % преобразование
            if Inew (i, j, k ) > 1
                Inew (i, j, k ) = 1;
            end
            if Inew (i ,j , k ) < 0
                Inew (i, j, k ) = 0;
            end
        end
    end
end

printComparison(I, Inew1, 'Растяжение динамического диапазона')

%% РАВНОМЕРНОЕ ПРЕОБРАЗОВАНИЕ

Inew2 = I;
for k = 1:1: Layers
    H = imhist(Inew2(:, :, k));
    CH = cumsum(H) ./ (numRows * numCols);  % нормированная кумулятивная гистограмма (для k-го слоя)
    Imin = min(min(I(: , : , k)));
    Imax = max(max(I(: , : , k)));
    for i = 1:1: numRows
        if Inew2(i,j,k) == 0
            Inew2(i, j, k) = 1;
        end
        for j = 1:1: numCols
            Inew2(i,j,k) = double(Imax - Imin) * (CH(I(i,j,k))) + Imin/255;   % преобразование
        end
    end
end

printComparison(I, Inew2, 'Равномерное преобразование')

%% ЭКСПОНЕНЦИАЛЬНОЕ ПРЕОБРАЗОВАНИЕ
I = imread("cat3.png");
Inew3 = double(I);
alfa = 5;
for k = 1:1: Layers
    H = imhist(I(:, :, k));
    CH = cumsum(H) ./ (numRows * numCols);  % нормированная кумулятивная гистограмма (для k-го слоя)
    Imin = min(min(I(: , : , k)));
    for i = 1:1: numRows
        for j = 1:1: numCols
            index = I(i,j,k);
            if index == 0
                index = 1;
            end
            Inew3(i,j,k) = double(Imin) / 256.0 - (1 / alfa) * log(1 - CH(index));   % преобразование
        end
    end
end

printComparison(I, Inew3, 'Экспоненциальное преобразование')

%% ПРЕОБРАЗОВАНИЕ ПО ЗАКОНУ РЭЛЕЯ
I = imread("cat2.png");
Inew4 = double(I);
alfa = 0.3;
for k = 1:1: Layers
    Imin = min(min(I(: , : , k)));
    H = imhist(I(:, :, k));
    CH = cumsum(H) ./ (numRows * numCols);  % нормированная кумулятивная гистограмма (для k-го слоя)
    for i = 1:1: numRows
        for j = 1:1: numCols
            index = I(i,j,k);
            if index == 0
                index = 1;
            end
            Inew4(i,j,k) = double(Imin)/256 + sqrt(2 * alfa^2 * log(1 / (1 - CH(index))));   % преобразование
        end
    end
end

printComparison(I, Inew4, 'Преобразование по закону Рэлея')

%% ПРЕОБРАЗОВАНИЕ ПО ЗАКОНУ СТЕПЕНИ 2/3

Inew5 = double(I);
for k = 1:1: Layers
    H = imhist(I(:, :, k));
    CH = cumsum(H) ./ (numRows * numCols);  % нормированная кумулятивная гистограмма (для k-го слоя)
    for i = 1:1: numRows
        for j = 1:1: numCols
            index = I(i,j,k);
            if index == 0
                index = 1;
            end
            Inew5(i,j,k) = CH(index)^(2/3);   % преобразование
        end
    end
end

printComparison(I, Inew5, 'Преобразование по закону степени 2/3')

%% ГИПЕРБОЛИЧЕСКОЕ ПРЕОБРАЗОВАНИЕ

Inew6 = double(I);
alfa = 0.04;
for k = 1:1: Layers
    H = imhist(I(:, :, k));
    CH = cumsum(H) ./ (numRows * numCols);  % нормированная кумулятивная гистограмма (для k-го слоя)
    for i = 1:1: numRows
        for j = 1:1: numCols
            index = I(i,j,k);
            if index == 0
                index = 1;
            end
            Inew6(i,j,k) = alfa ^ CH(index);   % преобразование
        end
    end
end

printComparison(I, Inew6, 'Гиперболическое преобразование')

%% ВСТРОЕННЫЕ ФУНКЦИИ MATLAB

Inew7 = I;
for k = 1:1: Layers
    Inew7(:, :, k) = imadjust(Inew7(:, :, k));
end

Inew8 = I;
for k = 1:1: Layers
    Inew8(:, :, k) = histeq(Inew8(:, :, k));
end

Inew9 = I;
for k = 1:1: Layers
    Inew9(:, :, k) = adapthisteq(Inew9(:, :, k));
end

printComparison(I, Inew7, 'Повышение контрастности (встроенная функция)')
printComparison(I, Inew8, 'Выравнивание гистограммы (встроенная функция)')
printComparison(I, Inew9, 'Контрастно-ограниченное адаптивное выравнивание гистограммы (встроенная функция)')

%% ПРОФИЛЬ ИЗОБРАЖЕНИЯ

C = imread('штрихкод.png');  % кружочек ещё надо
[numRows , numCols , Layers ] = size(C);
x = [1 numCols];
y = [ceil(numRows / 2) ceil(numRows / 2)];

figure('Name','Профиль изображения','NumberTitle','off');
subplot(2, 3, 2);
imshow(C);
for k = 1:1: Layers       % цикл для добавления subplot для профиля каждого цветового канала
    subplot(2, 3, k+3);
    improfile(C(:, :, k), x, y ),grid on;
end

%% ПРОЕКЦИЯ ИЗОБРАЖЕНИЯ
B = imread("сердце.png");
[numRows, numCols, Layers] = size(B);

% проекция на ось Oy
for i =1:1:numRows
    ProjOy(i, 1) = (round(sum(B(i, :, 1))) + round(sum(B(i, :, 2))) + round(sum(B(i, :, 3)))) / (256 * 3);
end

% проекция на ось Ox
for i =1:1:numCols
    ProjOx(1, i) = (round(sum(B(:, i, 1))) + round(sum(B(:, i, 2))) + round(sum(B(:, i, 3)))) / (256 * 3);
end

figure('Name','Проекции изображения','NumberTitle','off');

subplot(2,2,1)
imshow(B);
title("Исходное изображение");

subplot(2,2,2);
plot(ProjOy, 1:numRows);
title("Проекция на ось Oy");

subplot(2,2,3);
plot(1:numCols, ProjOx);
title("Проекция на ось Ox");

%% Красивый вывод сравнения картинок

function printComparison(I, Inew, name)   % сравнение изменённого изображения с оригиналом
    figure('Name',name,'NumberTitle','off');
    %_________________вывод оригинальной картинки______________________%
    subplot(2,4,1);
    imhist(I(:, :, 1));  % красный канал (оригинал)
    title("Red");
    subplot(2,4,2);
    imhist(I(:, :, 2));  % зелёный канал (оригинал)
    title("Green");
    subplot(2,4,3);
    imhist(I(:, :, 3));  % синий канал (оригинал)
    title("Blue");
    subplot(2,4,4);
    imshow(I);           % оригинальная картинка
    %_________________вывод изменённой картинки______________________%
    subplot(2,4,5);
    imhist(Inew(:, :, 1));  % красный канал (изменённая картинка)
    title("Red");
    subplot(2,4,6);
    imhist(Inew(:, :, 2));  % зелёный канал (изменённая картинка)
    title("Green");
    subplot(2,4,7);
    imhist(Inew(:, :, 3));  % синий канал (изменённая картинка)
    title("Blue");
    subplot(2,4,8);
    imshow(Inew);           % изменённая картинка
end