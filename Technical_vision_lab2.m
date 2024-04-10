clc
clear all
I = imread("cat2.png");

%% СДВИГ ИЗОБРАЖЕНИЯ

I = imread("cat2.png");
T = [1 0 0;
    0 1 0;
    50 100 1];   %% сдвиг по х на 50 пикселей, по у - на 100
tform = affine2d (T);
I_shift = imwarp(I, tform, ...
    'Interp', 'nearest', ...
    'OutputView', imref2d(size(I), ...
    [1 size(I,2)], [1 size(I,1)]));
printComparison(I, I_shift, "Сдвиг")

%% ОТРАЖЕНИЕ ИЗОБРАЖЕНИЯ

I = imread("cat2.png");
figure('Name',"Отражение относительно осей Ох и Оy",'NumberTitle','off');
subplot(1,3,1);
imshow(I);           % оригинальная картинка

T = [1 0 0;
    0 -1 0;
    0 0 1];
tform = affine2d(T);
I_reflect_x = imwarp(I, tform);
subplot(1,3,2);
imshow(I_reflect_x);           % отражение относительно Ох

T = [-1 0 0;
    0 1 0;
    0 0 1];
tform = affine2d(T);
I_reflect_y = imwarp(I, tform);
subplot(1,3,3);
imshow(I_reflect_y);           % отражение относительно Оу

%% ОДНОРОДНОЕ МАСШТАБИРОВАНИЕ (увеличение в 3 раза)

I = imread("cat2.png");
T = [3 0 0;     %% координата Х увеличивается в 3 раза
    0 3 0;      %% координата Y увеличивается в 3 раза
    0 0 1];
tform = affine2d(T);
I_scale = imwarp(I , tform);

figure('Name',"Исходное изображение",'NumberTitle','off');
imshow(I);

figure('Name',"Увеличенное изображение",'NumberTitle','off');
imshow(I_scale);


%% ПОВОРОТ НА 28° ПРОТИВ ЧАСОВОЙ СТРЕЛКИ

I = imread("cat2.png");
phi = 28 * pi /180;
T = [cos(phi) sin(phi) 0; 
    -sin(phi) cos(phi) 0;
    0 0 1];
tform = affine2d(T);
I_rotate = imwarp(I , tform);
printComparison(I, I_rotate, "Поворот на 28° против часовой стрелки")

%% СКОС ИЗОБРАЖЕНИЯ (s = 0.4)

I = imread("cat2.png");
T = [1 0 0;
    0.4 1 0;
    0 0 1];
tform = affine2d(T);
I_bevel = imwarp(I, tform);
printComparison(I, I_bevel, "Скос изображения (s=0.4)")

%% КУСОЧНО-ЛИНЕЙНОЕ ОТОБРАЖЕНИЕ

I = imread("cat2.png");
imid = round(size(I, 2) / 2);  %% середина по оси Ох
I_left = I(:, 1: imid, :);   %% кусок от начала до середины
stretch = 2;
I_right = I(:, (imid-1: end) , :);  %% кусок от середины до конца
T = [stretch 0 0;
    0 1 0;
    0 0 1];   %% растяжение по оси Ох
tform = affine2d(T);
I_scale = imwarp(I_right , tform);
I_piecewiselinear = [I_left I_scale];   %% обратно слепляем две половинки
printComparison(I, I_piecewiselinear, "Кусочно-линейное отображение")  %% получается пьяный кот


%% ПРОЕКТИВНОЕ ОТОБРАЖЕНИЕ

I = imread("cat2.png");
T = [1.08 0.25 0;
    0.6 1.1 0;
    0.075 0.005 1.03];
tform = projective2d (T);
I_projective = imwarp (I , tform );
printComparison(I, I_projective, "Проективное изображение")


%% ПОЛИНОМИАЛЬНОЕ ОТОБРАЖЕНИЕ

I = imread("cat2.png");
[numRows, numCols, Layers] = size(I);
T = [0 0; 1 0; 0 1;
    0.00001 0;
    0.002 0; 0.001 0];
for k = 1:1: Layers
    for y = 1:1: numCols
        for x =1:1: numRows
            xnew = round(T (1, 1)+ T (2, 1)* x +...
                T(3, 1)* y + T(4, 1)* x^2 +...
                T(5, 1)* x * y + T(6, 1)* y^2);
            ynew = round (T(1, 2)+ T(2, 2)* x +...
                T(3, 2)* y + T(4, 2)* x^2+...
                T(5, 2)* x * y + T(6, 2)* y^2);
            I_polynomial(xnew, ynew, k) = I(x, y, k);
        end
    end
end
printComparison(I, I_polynomial, "Полиномиальное отображение")

%% СИНУСОИДАЛЬНОЕ ОТОБРАЖЕНИЕ

I = imread("cat2.png");
[xi, yi] = meshgrid(1: numCols, 1: numRows);
imid = round(size(I, 2)/2);
u = xi + 20 * sin(2 * pi * yi / 90);
v = yi;
tmap_B = cat (3 ,u , v );
resamp = makeresampler ('linear', 'fill');
I_sinusoid = tformarray (I, [], resamp, [2 1], [1 2], [], tmap_B, 3);
printComparison(I, I_sinusoid, "Синусоидальное преобразование");

%% БОЧКООБРАЗНАЯ ДИСТОРСИИЯ

%%I = imread("cat2.png");
I = imread("pillow2.png");
[numRows, numCols, Layers] = size(I);
[xi , yi] = meshgrid (1: numCols ,1: numRows);
imid = round (size(I, 2)/2);
xt = xi (:) - imid;
yt = yi (:) - imid;
[theta, r] = cart2pol (xt, yt);
%%F3 = .00000001;  старые числа, делают норм бочку
%%F5 = .000000012;
F3 = .00001;
F5 = .000000006;
R = r + F3 * r .^3+ F5 * r .^5;
[ut, vt] = pol2cart(theta, R);
u = reshape(ut, size (xi)) + imid ;
v = reshape(vt, size (yi)) + imid ;
tmap_B = cat(3, u, v );
resamp = makeresampler ('linear', 'fill');
I_barrel = tformarray (I ,[] , resamp ,...
[2 1] ,[1 2] ,[] , tmap_B ,.3);
printComparison(I, I_barrel, "Бочкообразная дисторсия");


%% ПОДУШКООБРАЗНАЯ ДИСТОРСИЯ

%%I = imread("cat2.png");
I = imread("barrel2.png");
[numRows, numCols, Layers] = size(I);
[xi , yi] = meshgrid (1: numCols ,1: numRows);
imid = round (size(I, 2)/2);
xt = xi (:) - imid;
yt = yi (:) - imid;
[theta , r] = cart2pol (xt, yt);
F3 = -0.003;
R = r + F3 * r .^2;
[ut, vt] = pol2cart(theta, R);
u = reshape(ut, size (xi)) + imid ;
v = reshape(vt, size (yi)) + imid ;
tmap_B = cat(3, u, v );
resamp = makeresampler ('linear', 'fill');
I_pillow = tformarray (I ,[] , resamp ,...
[2 1] ,[1 2] ,[] , tmap_B ,.3);
printComparison(I, I_pillow, "Подушкообразная дисторсия");

%% СШИВКА ИЗОБРАЖЕНИЙ

topPart = imread('top.jpeg');
botPart = imread('bottom.jpeg');
topPartHT = im2double(rgb2gray(topPart));   %% переводим в ч/б
botPartHT = im2double(rgb2gray(botPart));
[numRows, numCols, Layers] = size(topPart);
[numRowsBot, numColsBot] = size(botPartHT);
intersecPart = 5;
botPartCorrHT = zeros(intersecPart, numCols);  %% создаём матрицы из нулей
topPartCorrHT = zeros(intersecPart, numCols);
correlationArray = [];

for j = 1:1: numCols        %% записываем в botPartCorrHT 5 первых строк нижнего изображения (для вычисления коэф-та корреляции со строками верхнего)
    for i = 1:1: intersecPart
        botPartCorrHT(i, j) = botPartHT(i, j);
    end
end

for j = 0:1: numRows - intersecPart
    for i = 1:1: intersecPart
        topPartCorrHT(i, :) = topPartHT(i +j, :);
    end
    correlationCoefficient = corr2(topPartCorrHT, botPartCorrHT);
    correlationArray = [correlationArray correlationCoefficient];
    correlationCoefficient = 0;
end
[M, I] = max(correlationArray);

numRowsBotCorr = numRowsBot + I - 1;
for k = 1:1: Layers                   %% склеиваем изображения
    for j = 1:1: numCols
        for i = 1:I - 1
            result_img(i, j, k) = topPart(i, j, k);  %% сначала записываем верхнее
        end
        for i = I :1: numRowsBotCorr
            result_img(i, j, k) = botPart(i - I + 1, j, k);  %% потом нижнее изображение, начиная с нужной строчки
        end
    end
end

figure('Name',"Верхняя часть Рейчел",'NumberTitle','off');
imshow(topPart);

figure('Name',"Нижняя часть Рейчел",'NumberTitle','off');
imshow(botPart);

figure('Name',"Сшитая Рейчел",'NumberTitle','off');
imshow(result_img);

%% Красивый вывод сравнения картинок

function printComparison(I, Inew, name)   % сравнение изменённого изображения с оригиналом
    figure('Name',name,'NumberTitle','off');
    subplot(1,2,1);
    imshow(I);           % оригинальная картинка

    subplot(1,2,2);
    imshow(Inew);           % изменённая картинка
end