function [sfondo, bianca, grigia,liquor,tutto] = Segmenter(matrice,background,whiteMatter,greyMatter,csf,max_iter,tol_abs,slice)


DIM = size(matrice);


% PREALLOCAZIONE DEI CLUSTER (DIM MASSIMA 91--> NUMERO IMMAGINI 2D)

cluster_1 = cell(DIM(3),1);      % 1 sfondo
cluster_2 = cell(DIM(3),1);      % 2 sostanza bianca
cluster_3 = cell(DIM(3),1);      % 3 sostanza grigia
cluster_4 = cell(DIM(3),1);      % 4 liquor

% max_iter = 500;  % iterazioni massime
% tol_abs = 0.05;  % soglia di arresto

% PREALLOCAZIONE CENTROIDI
% 1 219 131 62
centroid_1 = zeros(max_iter,1);
centroid_1(1) = background;                  % sfondo
centroid_2 = zeros(max_iter,1);
centroid_2(1) = whiteMatter;                % sostanza bianca
centroid_3 = zeros(max_iter,1);
centroid_3(1) = greyMatter;                % sostanza grigia
centroid_4 = zeros(max_iter,1);
centroid_4(1) = csf;                 % liquor


% inizializzazione degli elementi per il calcolo dei nuovi centroidi
% sommatorie dei livelli di grigio -> le pongo inizialmente a zero

Sgray1 = 0; 
Sgray2 = 0; 
Sgray3 = 0; 
Sgray4 = 0;

N1 = 0; N2 = 0; N3 = 0; N4 = 0;


for j=1:max_iter

    for i=1:DIM(3)
        
        % ESTRAGGO LA SLICE i-ESIMA
        img = matrice(:,:,i);

        % Calcolo delle distanze di ogni singolo voxel dai centroidi dei cluster
        dist_1 = abs(img-centroid_1(j));
        dist_2 = abs(img-centroid_2(j));
        dist_3 = abs(img-centroid_3(j));
        dist_4 = abs(img-centroid_4(j));
        
        % OGNI DISTANZA MISURA PIXEL PER PIXEL LE DISTANZE DAI VALORI DEI
        % CENTROIDI

        
        % LA DIST_MIN E' UNA MATRICE DELLE DIMENSIONI DELL'IMMAGINE CHE
        % PERO' CONTIENE TUTTE LE DISTANZE MINIME

        dist_min = min(min(min(dist_1,dist_2),dist_3),dist_4);

        % RICERCANDO GLI INDICI CORRISPONDENTI ALLA DISTANZA MINORE SI
        % RIESCE A SUDDIVIDERE GLI INDICI APPARTENENTI AI QUATTRO CLUSTERS

        cluster_1{i} = find(dist_1==dist_min); % ricavo i pixel più vicini al valore del primo cluster
        cluster_2{i} = find(dist_2==dist_min);
        cluster_3{i} = find(dist_3==dist_min);
        cluster_4{i} = find(dist_4==dist_min);

        
        % ESTRAGGO I VALORI EFFETTIVI DELL'IMMAGINE CORRISPONDENTI AGLI
        % INDICI DEL PRIMO CLUSTER
        
        m1 = img(cluster_1{i}); % estraggo il vettore relativo ai valori più vicini al valore del primo cluster

        % DAI VALORI EFFETTIVI SOMMO (CUMULATIVAMENTE) I LIVELLI DI GRIGIO
        % ESTRATTI
        Sgray1 = Sgray1 + sum(m1); % sommo tutti i livelli di grigio dei valori trovati 

        % SOMMO (CUMULATIVAMENTE) I PIXEL CORRISPONDENTI ALLE DISTANZE
        % MINIME DEL PRIMO CLUSTER

        N1 = N1 + length(m1); % sommo i numeri per trovare la quantità dei pixel




        m2 = img(cluster_2{i});
        Sgray2 = Sgray2 + sum(m2);
        N2 = N2 + length(m2);

        m3 = img(cluster_3{i});
        Sgray3 = Sgray3 + sum(m3);
        N3 = N3 + length(m3);

        m4 = img(cluster_4{i});
        Sgray4 = Sgray4 + sum(m4);
        N4 = N4 + length(m4);
        

    end % fine delle slice

    j=j+1;

    % OTTENENDO I VALORI MEDI DELLE DISTANZSE MINIME RICALCOLO I CENTROIDI
    centroid_1(j) = Sgray1/N1;
    centroid_2(j) = Sgray2/N2;
    centroid_3(j) = Sgray3/N3;
    centroid_4(j) = Sgray4/N4;

    % RIAZZERO I PARAMETRI PER LA PROSSIMA SLICE

    Sgray1 = 0; 
    Sgray2 = 0; 
    Sgray3 = 0; 
    Sgray4 = 0;

    N1 = 0; N2 = 0; N3 = 0; N4 = 0;


    % Se tutte le differenze fra i centroidi appena calcolati e i centroidi
    % precedenti sono inferiori alla soglia tol_abs allora l'algoritmo viene interrotto

    if abs(centroid_1(j)-centroid_1(j-1))<=tol_abs && abs(centroid_2(j)-centroid_2(j-1))<=tol_abs && ...
       abs(centroid_3(j)-centroid_3(j-1))<=tol_abs && abs(centroid_4(j)-centroid_4(j-1))<=tol_abs
        break;
    end

    
end 



% Fase di assemblaggio dei cluster

sfondo = [];
bianca = [];
grigia = [];
liquor= [];
cluster_all = [];

black = zeros(DIM(1:2));

for i=flip(1:(DIM(3)))

    img1 = zeros(DIM(1:2));
    img2 = zeros(DIM(1:2));
    img3 = zeros(DIM(1:2));
    img4 = zeros(DIM(1:2));

    % C = cat(dim,A,B) concatenates B to the end 
    % of A along dimension dim when A and B have compatible sizes 
    % 3 significa che è concatenata lungo la terza dimensione
 
    img1(cluster_1{i}) = 255; % parto dall'immagine nera e metto bianchi i pixel dei valori del primo cluster
    sfondo = cat(3,img1,sfondo); % unisco in dimensione a quelli prima
    img1 = cat(3,img1,img1,black); % variabile utilizzata per la visualizzazione totale

    img2(cluster_2{i}) = 212;
    bianca = cat(3,img2,bianca);
    img2 = cat(3,img2,black,black);

    img3(cluster_3{i}) = 170;
    grigia = cat(3,img3,grigia);
    img3 = cat(3,black,img3,black);

    img4(cluster_4{i}) = 128;
    liquor = cat(3,img4,liquor);
    img4 = cat(3,black,black,img4);

    imgtot = img1+img2+img3+img4;
    cluster_all = cat(4,imgtot,cluster_all);

end



tutto = bianca+grigia+liquor;

%%%%%%%%%%%%%%%%%%%%% VISUALIZZO I RISULTATI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



figure('Name','Originale')
sgtitle("Slice originaria e segmentata")
subplot(1,2,1)
imshow(uint8(squeeze(matrice(:,:,slice))))
title('Original')
subplot(1,2,2)
imshow(uint8(squeeze(cluster_all(:,:,:,slice))))
title('After segmentation')

figure('Name','Segmentazione')

sgtitle("Risultati per cluster")
subplot(2,2,1)

centroid_1 = centroid_1(centroid_1~=0);
centroid_2 = centroid_2(centroid_2~=0);
centroid_3 = centroid_3(centroid_3~=0);
centroid_4 = centroid_4(centroid_4~=0);

imshow(uint8(squeeze(sfondo(:,:,slice))),'Colormap',[0 0 0; 1 1 0])
title(['Background centroid value: ', num2str(centroid_1(end))])

subplot(2,2,2)
imshow(uint8(squeeze(bianca(:,:,slice))),'Colormap',[0 0 0; 1 0 0])
title(['White matter centroid value: ', num2str(centroid_2(end))])

subplot(2,2,3)
imshow(uint8(squeeze(grigia(:,:,slice))),'Colormap',[0 0 0; 0 1 0])
title(['Grey matter centroid value:', num2str(centroid_3(end))])

subplot(2,2,4)
imshow(uint8(squeeze(liquor(:,:,slice))),'Colormap',[0 0 0; 0 0 1])
title(['CSF centroid value: ' , num2str(centroid_4(end))])

figure('Name','Andamento')

sgtitle("Stabilizzazione valori centroidi")

subplot(2,2,1)
plot(1:length(centroid_1),centroid_1,'o');
yline(centroid_1(end),'r')
xlabel("Iterate")
ylabel("Background centroid value")
title('Background cluster')
grid on
grid minor

subplot(2,2,2)
plot(1:length(centroid_2),centroid_2,'o');
yline(centroid_2(end),'r')
xlabel("Iterate")
ylabel("White matter centroid value")
title('White matter cluster')
grid on
grid minor

subplot(2,2,3)
plot(1:length(centroid_3),centroid_3,'o');
yline(centroid_3(end),'r')
xlabel("Iterate")
ylabel("Grey matter centroid value")
title('Grey matter cluster')
grid on
grid minor

subplot(2,2,4)
plot(1:length(centroid_4),centroid_4,'o');
yline(centroid_4(end),'r')
xlabel("Iterate")
ylabel("CSF centroid value")
title('CSF cluster')
grid on
grid minor

end

