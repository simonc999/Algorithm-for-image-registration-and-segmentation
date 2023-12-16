% ESERCITAZIONE BIOIMMAGINI

% Esercitazione utilizzando l'immagine coronal2023
% Descrizione dei file:
% 1 ---------------------- coronal A.tif : immagine di riferimento
% 2 ---------------------- coronal B.tif : immagine rototraslata
% 3 ---------------------- coronal C.tif (Shear) doppio fattore di scala
% 4 ---------------------- coronal D.tif (Zoom out) singolo fattore di scala

%% PUNTO 1:

% Va creato un programma per visualizzare le immagini A, B, C, D con A di
% riferimento (immagine fissa), la loro diferenza e l'istogramma congiunto

clc
clear
close all
    
% imgReader('RM_Encefalo_A.tif','RM_Encefalo_B.tif','RM_Encefalo_C.tif','RM_Encefalo_D.tif');
% imgReader('coronal_A.tif','coronal_B.tif','coronal_C.tif','coronal_D.tif');
imgReader("coronal2023",'coronal A.tif','coronal B.tif','coronal C.tif','coronal D.tif');
% imgReader('rm_caviglia_A.tif','rm_caviglia_B.tif','rm_caviglia_C.tif','rm_caviglia_D.tif');

% imgReader('rm_caviglia_A.tif','rm_caviglia_B.tif');
% imgReader('coronal_A.tif','coronal_B.tif');
imgReader("coronal2023",'coronal A.tif','coronal B.tif');
% imgReader('rm_caviglia_A.tif','rm_caviglia_B.tif');
imgReader("coronal2023",'coronal A.tif','coronal A.tif');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%% METODO POINT-BASED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear
close all


%%%%%%%%%%%%%%%%%%%%%%%%% SCELTA IMMAGINE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% rif_img = imread('Rachide A.tif');
% float_img = imread('Rachide B.tif');
% RANGE = [10 20];

% rif_img = imread('RM_Encefalo_A.tif');
% float_img = imread('RM_Encefalo_B.tif');
% RANGE=[5 20];

% rif_img = imread('rm_caviglia_A.tif');
% rif_img = uint8((double(rif_img(:,:,1))+double(rif_img(:,:,2))+double(rif_img(:,:,3)))./(3));
% float_img = imread('rm_caviglia_B.tif');
% float_img = uint8((double(float_img(:,:,1))+double(float_img(:,:,2))+double(float_img(:,:,3)))./(3));
% RANGE=[3 12];

rif_img = imread('coronal A.tif');

float_img = imread('coronal B.tif');
ScaleFactor = false;

% float_img = imread('coronal D.tif');
% ScaleFactor = true;


RANGE = [10 20];

% rif_img = imread('coronal_A.tif');
% float_img = imread('coronal_B.tif');
% RANGE=[7 20];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
subplot(1,2,1)
imshow(rif_img)
title('Immagine riferimento')
subplot(1,2,2)
imshow(float_img)
title('Immagine da registrare')

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>> PREPROCESSING >>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

% >>>>>>>
% >>>>>>> 1 MASCHERA DI CONVOLUZIONE --------------------------------------
% >>>>>>>

% Creazione della maschera laplaciana e filtraggio delle due immagini
F = [-1 -1 -1; -1 8 -1; -1 -1 -1];
% F = [0 -1 0; -1 4 -1; 0 -1 0];
rif_img_laplace = imfilter(rif_img,F);
float_img_laplace = imfilter(float_img,F);

figure
subplot(2,3,1)
imshow(rif_img_laplace)
title('Immagine riferimento con filtro laplaciano')
subplot(2,3,4)
imshow(float_img_laplace)
title('Immagine da registrare con filtro laplaciano')


% >>>>>>>
% >>>>>>> 2 RIEMPIMENTO DEI BUCHI -----------------------------------------
% >>>>>>>

rif_img_fill = imfill(rif_img_laplace,'holes');
float_img_fill = imfill(float_img_laplace,'holes');

subplot(2,3,2)
imshow(rif_img_fill)
title('Immagine riferimento con filtro laplaciano (buchi riempiti)')
subplot(2,3,5)
imshow(float_img_fill)
title('Immagine da registrare con filtro laplaciano (buchi riempiti)')

% >>>>>>>
% >>>>>>> 3 BINARIZZAZIONE ------------------------------------------------
% >>>>>>>

rif_img_bin = imbinarize(rif_img_fill,0.3);
float_img_bin = imbinarize(float_img_fill,0.3);

subplot(2,3,3)
imshow(rif_img_bin)
title('Laplaciano immagine riferimento (binaria)')
subplot(2,3,6)
imshow(float_img_bin)
title('Laplaciano immagine da registrare (binaria)')

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>> RICERCA DEI MARKERS >>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% Ricerca dei centri dei cerchi nelle due immagini

[markerA, raggioA, coordA] = imfindcircles(rif_img_bin,RANGE);
[markerB, raggioB, coordB] = imfindcircles(float_img_bin,RANGE);


markerA = markerA(1:3,:);
raggioA = raggioA(1:3,:);
coordA = coordA(1:3,:);

markerB = markerB(1:3,:);
raggioB = raggioB(1:3,:);
coordB = coordB(1:3,:);


figure
subplot(1,2,1)
imshow(rif_img)
viscircles(markerA, raggioA,'EdgeColor','g');
title('Markers identificati (immagine riferimento)')
subplot(1,2,2)
imshow(float_img)
viscircles(markerB, raggioB,'EdgeColor','g');
title('Markers identificati (immagine da registrare)')

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>> RICERCA DEI CENTROIDI >>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>> CONVERSIONE DI COORDINATE >>>>>>>>>>>>>>>>>>>>>>

pixel = size(rif_img_bin);
markerA_PLOT(:,1) = markerA(:,1) - floor(pixel(2)/2);
markerA_PLOT(:,2) = floor(pixel(1)/2) - markerA(:,2);

markerB_PLOT(:,1) = markerB(:,1) - floor(pixel(2)/2);
markerB_PLOT(:,2) = floor(pixel(1)/2) - markerB(:,2);

ca_plot= centroidCustom(markerA_PLOT(:,1),markerA_PLOT(:,2));

cb_plot= centroidCustom(markerB_PLOT(:,1),markerB_PLOT(:,2));


figure('Name','Posizione dei marker e dei centroidi')
subplot(1,2,1)
I1 = imshow(rif_img);
I1.AlphaData = 0.4;
hold on 
plot(markerA_PLOT(:,1)+floor(pixel(2)/2) ,floor(pixel(1)/2) -markerA_PLOT(:,2),'g+',LineWidth = 3)
hold on  
plot(ca_plot(:,1) +floor(pixel(2)/2) ,floor(pixel(1)/2)-ca_plot(:,2),'r*',LineWidth = 5)
title('Immagine di riferimento')
subplot(1,2,2)
I2 = imshow(float_img);
I2.AlphaData = 0.4;
hold on 
plot(markerB_PLOT(:,1)+floor(pixel(2)/2) ,floor(pixel(1)/2) -markerB_PLOT(:,2),'g+',LineWidth = 3)
hold on 
plot(cb_plot(:,1)+floor(pixel(2)/2) ,floor(pixel(1)/2)-cb_plot(:,2),'r*',LineWidth = 5)
title('Immagine da rototraslare')

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>> RICERCA DELLE CORRISPONDENZE >>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

dist=zeros(3,2);
   for i=1:length(markerA)
       for j=1:length(markerB)
           dist(i,j)=norm(markerA(i,:)-markerB(j,:)); %In ogni riga il marker di A è fisso
       end
   end

 [min_dist, in] = min(dist,[],2);


 markerB=markerB(in',:);

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>> INIZIALIZZO I PESI >>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


w = [1 1 1] ;

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>> Applico funzione di ricerca >>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

[alpha,tx,ty,s] = PointBasedFun(w,markerA,markerB,pixel,ScaleFactor);
 


img_rot = optimized_affine_trasformation(float_img,[alpha,tx,ty,s,s,0]);
immagine_riferimento = rif_img;
immagine_registrata = img_rot;
imgReader("Metodo centroidi (point-based)",immagine_riferimento,immagine_registrata);

theta = [alpha tx ty s s 0];
disp('--------------------------------------------------------------')
disp('    alpha     tx        ty          s         s          0')
disp(theta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%% METODO PAR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
close all
A = imread("coronal A.tif","tif");
B = imread("coronal B.tif","tif");
% B = imread("coronal D.tif","tif");
a = double(imread("coronal_A_bin.tif","tif")/255);
b = double(imread("coronal_B_bin.tif","tif")/255);
% b = double(imread("coronal_D_bin.tif","tif")/255);

% passo al metodo le immagini binarizzate
[I_A, I_B, xg_A, xg_B, yg_A, yg_B, X_P, Y_P] = ParMethodFun(a,b);

figure(5)
sgtitle("Visualizzazione immagini metodo PAR")
subplot(2,2,1)
imshow(A); 
title('coronal_A','Interpreter','none')
subplot(2,2,2)
imshow(a); 
title('coronal_A BINARY','Interpreter','none')
hold on 
plot(xg_A ,yg_A,'ro',LineWidth = 2)
hold on 
plot(xg_A ,yg_A,'g+',LineWidth = 1)

subplot(2,2,3)
imshow(B);
title('coronal_D','Interpreter','none')
subplot(2,2,4)
imshow(b); 
title('coronal_D BINARY','Interpreter','none')
hold on 
plot(xg_B ,yg_B,'ro',LineWidth = 2)
hold on 
plot(xg_B ,yg_B,'g+',LineWidth = 1)



% Autovettori
[autovet_A, ~] = eig(I_A);
[autovet_B, ~] = eig(I_B);
E1 = zeros(2,2);
E2 = E1;

for i=1:size(autovet_A,2)
    E1(:,i) = autovet_A(:,i)/norm(autovet_A(:,i),2);
    E2(:,i) = autovet_B(:,i)/norm(autovet_B(:,i),2);
end


ta = [X_P(end)/2 - xg_A ; Y_P(end)/2 - yg_A];
tb = [X_P(end)/2 - xg_B ; Y_P(end)/2 - yg_B];
Fs =round(sqrt(sum(sum(A))/sum(sum(B))),2);

p_A = [asin(E1(2,1)), ta(1), ta(2), 1, 1, 0]; % trovo un solo angolo
p_B = [asin(E2(2,1)), tb(1), tb(2), 1, 1, 0];
% p_B = [asin(autovet_B(2,1)), tb(1), tb(2), Fs, Fs, 0]; % per scala

[img_rot_A] = optimized_affine_trasformation(A,p_A);
[img_rot_B] = optimized_affine_trasformation(B,p_B);

disp('----------------- parameters for reference image -----------------')
disp('    alpha     tx        ty          s1         s2          s12')
disp(p_A)

disp('------------------- parameters for float image -------------------')
disp('    alpha     tx        ty          s1         s2          s12')
disp(p_B)

immagine_riferimento = img_rot_A;
immagine_registrata = img_rot_B;
imgReader("Metodo PAR",immagine_riferimento,immagine_registrata);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%% OTTIMIZZAZIONE DEI PARAMETRI %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc
close all

% UTILIZZO IL METODO DELLA OTTIMIZZAZIONE SLIDE 41/59 L8_2023_Registration

% I marker dell'immagine B sono le caratteristiche (features) dell'immagine
% che si muove, mentre i marker ddell'immagine A sono le caratteristiche
% dell'immagine fissa o di riferimento.

% Il problema della registrazione consiste nel trovare le corrispondenze 
% fra caratteristiche e nello stimare i parametri delle trasformazioni 
% basate su queste caratteristiche.La funzione obbiettivo dipende dai 
% parametri incogniti della trasformazione e dalle corrispondenze delle 
% features incognite (espresse come vincolo)

% I VALORI INCOGNITI SONO I PARAMETRI
% --> a (cos(alpha))
% --> b (sen(alpha))
% --> tx (traslazione lungo x)
% --> ty (traslazione lungo y)

% Calcolo le corrispondenze e scrivo una funzione obbiettivo E con
% parametri a b tx e ty incogniti. La soluzione sarà il punto in cui le
% derivate si annullano.
% in questo caso consideriamo la distanza tra i marker dell'immagine A e B.
% preistanzio i vettori degli indici di corrispondenze e delle distanze

% FILTER 1
 F = [-1 -1 -1; -1 8 -1; -1 -1 -1];

% FILTER 2
% F = [0 -1 0; -1 4 -1; 0 -1 0];

% %----------- RM_Encefalo.tif
% rif_img=imread('RM_Encefalo_A.tif');
% float_img=imread('RM_Encefalo_B.tif');
% %---RANGE FIRST FILTER
% %RANGE = [3 8];
% %---RANGE SECOND FILTER
% RANGE =[5 15];


% %----------- coronal.tif
% rif_img=imread('coronal_A.tif');
% float_img=imread('coronal_B.tif');
% %---FIRST FILTER
% % RANGE = [10 20];
% %---SECOND FILTER
% % RANGE = [5 17];

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>> CARICO L'IMMAGINE >>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

%----------- coronal_2023.tif
rif_img=imread('coronal A.tif');

float_img=imread('coronal B.tif');
sxx = 1; syy = 1; sxy = 0;

% float_img=imread('coronal D.tif');
% sxx = 1.0776; syy = 1.0776; sxy = 0;

RANGE = [10 20];


% %----------- rm_caviglia.tif
% rif_img = imread('rm_caviglia_A.tif');
% rif_img = uint8((double(rif_img(:,:,1))+double(rif_img(:,:,2))+double(rif_img(:,:,3)))./(3));
% float_img = imread('rm_caviglia_B.tif');
% float_img = uint8((double(float_img(:,:,1))+double(float_img(:,:,2))+double(float_img(:,:,3)))./(3));
% RANGE = [5 12];


% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>> TROVO I MARKERS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


[markerA, ~] = imfindcircles(imbinarize(imfill(imfilter(rif_img,F),'holes'),0.45),RANGE);
[markerB, ~] = imfindcircles(imbinarize(imfill(imfilter(float_img,F),'holes'),0.45),RANGE);

pixel = size(rif_img);

markerA(:,1) = markerA(:,1) - floor(pixel(2)/2);
markerA(:,2) = floor(pixel(1)/2) - markerA(:,2);

markerB(:,1) = markerB(:,1) - floor(pixel(2)/2);
markerB(:,2) = floor(pixel(1)/2) - markerB(:,2);

dist = zeros(height(markerB),1);
indice_dist_min = zeros(height(markerA),1);

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>> TROVO LE CORRISPONDENZE >>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

for i=1:height(markerA)
    
    for j=1:height(markerB)

        dist(j) = sqrt((markerA(i,1)-markerB(j,1))^2 + (markerA(i,2)-markerB(j,2))^2);

    end
    [~,indice_dist_min(i)] = min(dist);
end

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>> CALCOLO LA MATRICE DELLE CORRISPONDENZE >>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

% Matrice delle corrispondenze e calcolo dei parametri della trasformazione

% --          [M1_A(x) M1_A(y) M1_B(x) M1_B(y)
% --          M2_A(x) M2_A(y) M2_B(x) M2_B(y)
% --          M3_A(x) M3_A(y) M3_B(x) M3_B(y)]

mat_corr = [markerA(1,:),markerB(indice_dist_min(1),:); ...
            markerA(2,:),markerB(indice_dist_min(2),:);...
            markerA(3,:),markerB(indice_dist_min(3),:)];

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>> CALCOLO LA MATRICE X >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


% OTTENENDO LA FUNZIONE OBBIETTIVO E PONENDO LE DERIVATE RISPETTO AI
% PARAMETRI a b tx e ty OTTENIAMO LA MATRICE X

%     a          b         tx         ty

% indici 3  e 4 estraggo soltanto le componenti markers float xi e yi

X = [sum(mat_corr(:,3).^2 + mat_corr(:,4).^2), 0, sum(mat_corr(:,3)), sum(mat_corr(:,4));... % da = 0
     0, sum(mat_corr(:,3).^2 + mat_corr(:,4).^2), sum(-mat_corr(:,4)), sum(mat_corr(:,3));...% db = 0
     sum(mat_corr(:,3)), sum(-mat_corr(:,4)), 3, 0;... % dtx = 0
     sum(mat_corr(:,4)), sum(mat_corr(:,3)), 0, 3]; % dty = 0

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>> CALCOLO IL VETTORE b >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

% b E' IL VETTORE DEI TERMINI NOTI (dopo aver posto le derivate uguali a 0
% ci saranno dei valori a dx dell'uguale)

b = [sum(mat_corr(:,1).*mat_corr(:,3)+mat_corr(:,2).*mat_corr(:,4));...
    sum(mat_corr(:,3).*mat_corr(:,2)-mat_corr(:,4).*mat_corr(:,1));...
    sum(mat_corr(:,1)); sum(mat_corr(:,2))];

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>> CALCOLO I PARAMETRI THETA >>>>>>>>>>>>>>>>>>>>>>>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

% Questo è un sistema del tipo Xθ=b, se X è invertibile si ricavano a, b, 
% tx e ty.

theta = X\b;

% -- theta(1) = cos(alpha)
% -- theta(2) = -sen(alpha)
% -- theta(3) = tx
% -- theta(4) = -ty

% Trasformazione affine della float_img

% theta(1) = coseno
% theta(2) = seno

alpha = -atan(theta(2)/theta(1)); % il sistema di riferimento è opposto
tx = theta(3); 
ty = -theta(4); % il sistema di riferimento è opposto

% poniamo i fattori di scala pari a 1 ed 1 e il correlato pari a 0


img_rot = optimized_affine_trasformation(float_img,[alpha,tx,ty,sxx,syy,sxy]);

theta = [alpha tx ty sxx syy 0];
disp('--------------------------------------------------------------')
disp('    alpha     tx        ty          s1         s2          0')
disp(theta)

immagine_riferimento = rif_img;
immagine_registrata = img_rot;
imgReader("Metodo ottimizzazione dei parametri",immagine_riferimento,immagine_registrata);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%% INTENSITY BASED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc
close all


disp("1. SSD ")
disp("2. CC")
disp("3. RIU ")
disp("4. MI")


prompt = "Select a method: ";
x = input(prompt);


%%--- selezione immagine --------------------------------------------------

% %%----------- RM_Encefalo.tif
% forTitle = 'RM_Encefalo.tif';
% rif_img = imread('RM_Encefalo_A.tif');
% float_img = imread('RM_Encefalo_D.tif');
% % I VALORI VERI DELLA TRASFORMAZIONE SONO +8.11(X) +14.31(Y) E +8° SCALA X = SCALA Y = 1.064
% % time estimated for these ranges 4.178345 seconds.
% alfa = linspace(deg2rad(7),deg2rad(9),3); 
% trasx = linspace(8.109,8.11,2);
% trasy = linspace(14.3,14.4,2);
% scalex = linspace(1.05,1.064,2);
% scaley = linspace(1.05,1.064,2);

% ----------- coronal.tif
% forTitle = 'coronal.tif';
% rif_img = imread('coronal_A.tif');
% float_img = imread('coronal_D.tif');
% % Founded trasformation with traslation x,y : 14.2753, -6.5 pixels, a rotation of -8° , and a scale(x,y) factor of 1.125,1.125.
% % time estimated for these ranges 8.889971 seconds.
% alfa = linspace(deg2rad(-9),deg2rad(-7),3); 
% trasx = linspace(14.26,14.28,3);
% trasy = linspace(-6.5,-6.4,2);
% scalex = linspace(1.125,1.126,2);
% scaley = linspace(1.125,1.126,2);
% % 

%----------- coronal_2023.tif
forTitle = 'coronal_2023.tif';
rif_img = imread('coronal A.tif');
float_img = imread('coronal D.tif');
% Founded trasformation with traslation x,y : 7.9889, -10.8111 pixels, a rotation of -12° , and a scale(x,y) fator of 1.075,1.075.
% time estimated for these ranges 1.468795 seconds.
alfa = linspace(deg2rad(-13),deg2rad(-12),2); 
trasx = linspace(7,8,2);
trasy = linspace(-10.811,-10.8,2);
scalex = linspace(1.075,1.08,2);
scaley = linspace(1.075,1.08,2);
% float_img = imread('coronal C.tif');
% alfa = -11.6*pi/180 :0.1*pi/180:-11.5*pi/180;
% trasx = -4.9:0.1:-4.8;
% trasy= -10.6:0.1:-10.5;
% scalex = 1.07:0.01:1.08;
% scaley = 0.985:0.005:0.99;

% %----------- rm caviglia.tif (A  e B a 24 bit)
% forTitle = 'caviglia.tif';
% rif_img = imread('rm_caviglia_A.tif');
% rif_img = uint8((double(rif_img(:,:,1))+double(rif_img(:,:,2))+double(rif_img(:,:,3)))./(3));
% float_img = imread('rm_caviglia_D.tif');
% % Founded trasformation with traslation x,y : 27.81, 10.47 pixels, a rotation of 12° , and a scale(x,y) factor of 1.073,1.073.
% % % time estimated for these ranges 2.756350 seconds.
% alfa = linspace(deg2rad(11),deg2rad(13),3); 
% trasx = linspace(27.80,27.81,2);
% trasy = linspace(10.45,10.47,3);
% scalex = linspace(1.072,1.073,2);
% scaley = linspace(1.072,1.073,2);

%%-------------------------------------------------------------------------

dim = size(rif_img); 

% PREISTANZIO LA MATRICE DEGLI ERRORI

SSD = zeros(length(alfa),length(trasx),length(trasy),length(scalex),length(scaley));
CC = zeros(length(alfa),length(trasx),length(trasy),length(scalex),length(scaley));
RIU = zeros(length(alfa),length(trasx),length(trasy),length(scalex),length(scaley));
MI = zeros(length(alfa),length(trasx),length(trasy),length(scalex),length(scaley));

% IL CICLO SERVE PER POPOLARE L'ERRORE

tic
for  i=1:length(alfa) % //////////////// CICLO SUI GRADI

    for j=1:length(trasx) % //////////////// CICLO SULLE TRASLAZIONI IN X

        for k=1:length(trasy) % //////////////// CICLO SULLE TRASLAZIONI IN Y

            for u=1:length(scalex) % //////////////// CICLO SULLA SCALA IN X

                for v=1:length(scaley) % //////////////// CICLO SULLA SCALA IN Y


                    [img_rot]=optimized_affine_trasformation(float_img,[alfa(i) trasx(j) trasy(k) scalex(u) scaley(v) 0]);

                    % MASCHERA PER CONSIDERARE I PIXEL INTERSEZIONE
                    mask = intersectionMask(alfa(i),trasx(j),trasy(k),scalex(u),scaley(v),0,dim);

                    
                    switch x
                        case 1
                            
                            % Errore nella regione estratta dalla machera
                            img_err_inters = double(rif_img - img_rot).*mask;

                            % Calcolo della SSD
                            SSD(i,j,k,u,v) = sum(sum(img_err_inters.^2,'omitnan'),'omitnan');
                            titleString = 'SSD';
                        case 2 
                            img_rif_inters = double(rif_img).*mask;
                            img_rot_inters = double(img_rot).*mask;
        
                            %Calcolo dell'intesità media delle due immagini
                            mean_int_rif = mean(img_rif_inters(:),'omitnan');
                            mean_int_rot = mean(img_rot_inters(:),'omitnan');
        
                            %Calcolo del coefficiente CC
                            numer_CC = sum(sum((img_rif_inters-mean_int_rif).*(img_rot_inters-mean_int_rot),'omitnan'),'omitnan');
                            denom_CC = sqrt(sum(sum((img_rif_inters-mean_int_rif).^2,'omitnan'),'omitnan')*sum(sum((img_rot_inters-mean_int_rot).^2,'omitnan'),'omitnan'));
                            CC(i,j,k,u,v) = numer_CC/denom_CC;

                            titleString = 'Cross correlazione';
                        case 3
                            img_rif_inters = (double(rif_img)+1).*mask;
                            img_rot_inters = (double(img_rot)+1).*mask;
            
                            % Calcolo dell'immagine rapporto
                            ratio_img = img_rot_inters(:)./img_rif_inters(:);
            
                            % Calcolo della media immagine rapporto
                            mean_ratio_img = mean(ratio_img,'omitnan'); % TOGLIE IL NAN
            
                            % Calcolo della deviazione standard immagine rapporto
                            sd_ratio_img = std(ratio_img,'omitnan');
            
                            % Calcolo del coefficiente RIU
                            RIU(i,j,k,u,v) = sd_ratio_img/mean_ratio_img; % CV

                            titleString = 'RIU';


                        case 4
                            img_rif_inters = double(rif_img).*mask;
                            img_rot_inters = double(img_rot).*mask;

                            img_rif_inters(isnan(img_rif_inters)) = 0;
                            img_rot_inters(isnan(img_rot_inters)) = 0;
        
                            % Calcolo della probabilità con istogramma congiunto delle due immagini
                            [p_cong,p_marg_rif,p_marg_rot] = bihist(img_rif_inters,img_rot_inters);
        
                            % Calcolo delle entropie
                            E_rif = sum(p_marg_rif.*log2(p_marg_rif),'omitnan');
                            E_rot = sum(p_marg_rot.*log2(p_marg_rot),'omitnan');


                            E_cong = sum(p_cong(:).*log2(p_cong(:)),'omitnan');
        
                            % Calcolo della MI
                            MI(i,j,k,u,v) = (E_rif+E_rot)/E_cong; %normalizzata
                            % MI(i,j,k,u,v) = (E_rif+E_rot)-E_cong; %teorica

                            titleString = 'Mutua informazione';
                    end
                    
                end
            end
            
        end
    end
end
toc




if (x==1)
    [~,ind] = min(SSD(:));
elseif(x==2)
    [~,ind] = max(CC(:));
elseif(x==3)
    [~,ind] = min(RIU(:));
elseif(x==4)
    [~,ind] = max(MI(:));
else
    
end

    

% TROVO LE COORDINATE SPAZIALI DELLA POSIZIONE DEGLI INDICI 

[ind_alfa,ind_tx,ind_ty,ind_sx,ind_sy] = ind2sub(size(SSD), ind);
% RIFACCIO LA TRASFORMAZIONE CON LE COORDINATE CHE PORTANO A MINIMIZZARE
% L'ERRORE

img_rot = optimized_affine_trasformation(float_img,[alfa(ind_alfa),trasx(ind_tx),trasy(ind_ty),scalex(ind_sx),scaley(ind_sy),0]);

figure ('name', 'differenza e output')
sgtitle([titleString ' ' forTitle ], 'Interpreter', 'none')
subplot 221;imshow(rif_img);
title('Immagine di riferimento')
subplot 222;imshow(float_img);
title('Immagine float')
subplot 223;imshow(rif_img-img_rot);
title('Differenza tra le due immagini')
subplot 224;imshow (img_rot);
title('Immagine registrata')

 [p_cong, ~, ~] = bihist(rif_img, float_img);  % ISTOGRAMMA PRIMA DELLA TRASFORMAZIONE 
[p_cong_after, ~, ~] = bihist(rif_img, img_rot);  % ISTOGRAMMA DOPO LA TRASFORMAZIONE



figure('name', 'istogrammi di correlazione immagini')
sgtitle([titleString ' ' forTitle ], 'Interpreter', 'none')
subplot(2,1,1);
imshow(imadjust(p_cong));
title('Immagine di riferimento vs immagine da registrare')
subplot(2,1,2);
imshow(imadjust(p_cong_after));
title('Immagine di riferimento vs immagine registrata')

X = '--- Founded trasformation: ---';
X0 = [' a traslation x,y : ', num2str(trasx(ind_tx)),', ' num2str(trasy(ind_ty)),  ' pixels' ];
X1 = [' a rotation of ',num2str(rad2deg(alfa(ind_alfa))), '°'];
X2 = [' a scale(x,y) factor of ', num2str(scalex(ind_sx)), ',' , num2str(scaley(ind_sy))];
disp(X);
disp(X0);
disp(X1);
disp(X2);
