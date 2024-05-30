clc
clear all


%%  ДИЛАТАЦИЯ

I = imread("1.png");
binary_image = im2bw(I); % Преобразование изображения в бинарное
SE = strel('disk', 10); % создание структурного элемента (диск радиусом 10 пикселей)
dilated_image = imdilate(binary_image, SE); % применение операции дилатации
printComparison(I, dilated_image, "Дилатация")


%%  ЭРОЗИЯ

I = imread("1.png");
binary_image = im2bw(I);  % Преобразование изображения в бинарное
SE = strel('disk', 10); % создание структурного элемента (диск радиусом 10 пикселей)
eroded_image = imerode(binary_image, SE); % применение операции эрозии
printComparison(I, eroded_image, "Эрозия")

%% ОТКРЫТИЕ

I = imread("1.png");
binary_image = im2bw(I); % Преобразование изображения в бинарное
SE = strel('disk', 10); % создание структурного элемента (диск радиусом 10 пикселей)
opened_img = imopen(binary_image, SE); % применение операции открытия
printComparison(I, opened_img, "Открытие")

%% ЗАКРЫТИЕ

I = imread("1.png");
%%binary_image = im2bw(I); % Преобразование изображения в бинарное
SE = strel('disk', 10); % создание структурного элемента (диск радиусом 10 пикселей)
closed_img = imclose(binary_image, SE); % Применение операции закрытия
printComparison(I, closed_img, "Закрытие")

%% РАЗДЕЛЕНИЕ "СКЛЕЕННЫХ" ОБЪЕКТОВ

I = imread("склеенные.png");
t = graythresh(I);
Inew = im2bw(I, t);
Inew = ~ Inew;
BW2 = bwmorph(Inew, 'erode', 11);
BW2 = bwmorph(BW2, 'thicken', Inf);
Inew = ~(Inew & BW2);
printComparison(I, Inew, "Разделение склеенных объектов")

contour = bwperim(Inew);
printComparison(I, contour, "Разделение склеенных объектов (с контурами)")


%% СЕГМЕНТАЦИЯ МЕТОДОМ УПРАВЛЯЕМОГО ВОДОРАЗДЕЛА

rgb = imread('5.jpg');
A = rgb2gray(rgb);
B = strel('disk', 6);
C = imerode(A, B);
Cr = imreconstruct(C, A);
Crd = imdilate(Cr, B);
Crdr = imreconstruct(imcomplement(Crd), imcomplement(Cr));
Crdr = imcomplement(Crdr);
fgm = imregionalmax (Crdr);
A2 = A;
A2(fgm) = 255;
B2 = strel (ones(5, 5));
fgm = imclose(fgm, B2);
fgm = imerode(fgm, B2);
fgm = bwareaopen (fgm, 20);
A3 = A ;
A3(fgm) = 255;
bw = imbinarize(Crdr);
D = bwdist(bw);
L = watershed (D);
bgm = L == 0;
hy = fspecial ('sobel');
hx = hy';
Ay = imfilter(double(A), hy, 'replicate');
Ax = imfilter(double(A), hx, 'replicate');
grad = sqrt ( Ax .^2 + Ay .^2);
grad = imimposemin(grad, bgm | fgm);
L = watershed(grad);
A4 = A ;
A4 ( imdilate ( L == 0 , ones (3 ,3)) | bgm | fgm ) = 255;
Lrgb = label2rgb (L , 'jet', 'w', 'shuffle');
printComparison(rgb, Lrgb, "Сегментация методом управляемого водораздела")


%% Красивый вывод сравнения картинок

function printComparison(I, Inew, name)   % сравнение изменённого изображения с оригиналом
    figure('Name',name,'NumberTitle','off');
    subplot(1,2,1);
    imshow(I);           % оригинальная картинка

    subplot(1,2,2);
    imshow(Inew);           % изменённая картинка
end