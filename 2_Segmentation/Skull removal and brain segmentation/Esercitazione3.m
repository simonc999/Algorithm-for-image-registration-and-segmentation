clear
close all
clc

% PREALLOCAZIONE VOLUME MASCHERA
VOLUMEmask = false(156,206,155);


T3D = load_untouch_nii('T13D_original_clean.nii');
immagine = T3D.img;

DIM = size(immagine);


general = figure("Name", "Process");
background = figure("Name", "Contour");


SHIFTING_INDEX = 0;

indexing_visualize = [10 90 140];
%indexing_visualize = 90:5:135;
sizeSubplot = 7;

for I = 1:DIM(3)
    sgtitle("Preprocessing slice " + num2str(I) + "/155...")
    % PREALLOCATING THE PATH MATRIX
    matrixDraw = false(DIM(1),DIM(2));

    % PREALLOCATING THE PATH MATRIX FILLED
    matrixDrawFILLED = false(DIM(1),DIM(2));

    % subplot(3,3,I) % UNCOMMENT FOR VISUALIZE
    
    index = SHIFTING_INDEX + I;  % SHIF SUBPLOT FOR VISUALIZE

    std =1;  % STANDARD DEVIATION GAUSSIAN FILTER
    blacklv = -110; % LEVEL FOR INDEX SEARCH
    flagSEARCH = true; % BOOLEAN FOR ERASING SLICE (true = draw, false = skip)

    if index>=5 && index<16 % ------------------------AUTO
        checkLINE = 63;
        std = 0.9;

    elseif index>=16 && index<25  % MANUALLY CORRECT AFTER THE CYCLE
        blacklv = -90;
        checkLINE = 63;
        std = 0.5;

    elseif index>=25 && index<31
        blacklv = -90;
        std = 0.9;
        checkLINE = 57;

    elseif index>=31 && index<43
        blacklv = -98;
        std = 1.2;
        checkLINE = 57;

    elseif index>=43 && index<52
        blacklv = -100;
        std = 0.1;
        checkLINE = 63;

    elseif index>=52 && index<69
        blacklv = -100;
        std = 0.3;
        checkLINE = 63;

    elseif index>=69 && index<127
        checkLINE = 70;
        elseif index >= 127 && index<139
        checkLINE = 140;

    elseif index>=139 && index<149
        checkLINE = 110;

    else
        flagSEARCH = false;
    end

    
    if(flagSEARCH) 

        slice = immagine(:,:,index); % EXTRACTING THE SLICE             

        if(ismember(I,indexing_visualize)) %%%%%%%%%%%% v %%%%%%%%%%%%%%%%%
            figure(general)
            indexDRAW = ((find(indexing_visualize == I)-1)*sizeSubplot)+1;
            subplot(length(indexing_visualize),sizeSubplot,indexDRAW)

            surfc(-slice)
            title(["Surface of negative " num2str(I)])

            subplot(length(indexing_visualize),sizeSubplot,indexDRAW+1)

            Im = imshow(uint8(squeeze(immagine(:,:,I))));
            Im.AlphaData = 0.8;
            hold on
            contour(-slice)
            colormap(gca, 'default');
            title("Contouring")
            colorbar('Location', 'eastoutside');

            subplot(length(indexing_visualize),sizeSubplot,indexDRAW+2)
            contour(imgaussfilt(-slice,std));
            title(["Gaussian filtering std = " num2str(std)])

            subplot(length(indexing_visualize),sizeSubplot,indexDRAW+3)
            [M,c] = contour(imgaussfilt(-slice,std), [blacklv blacklv] );
            title(["Extracting level " num2str(blacklv)])
        else
            figure(background)
            M = contour(imgaussfilt(-slice,std), [blacklv blacklv] );        
            
        end %%%%%%%%%%%%%%%%%%%%%%%%%%%%% A %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        matrixtest = false(156,206);
        
        % CLEANING CONTOUR POINT

        indexesX = (M(1,:)<0 | M(1,:)> DIM(2));
        indexesY = (M(2,:)>DIM(1) | M(2,:)<0);
        INDEXES = indexesY | indexesX;
        M(:, INDEXES) = [];

        % DISCRETIZING

        M = round(M);
        

        % DRAWING CONTOUR
        
        for i=1:length(M)
            matrixtest(M(2,i),M(1,i))=true;
        end

        % EXTRACTING START INDEX

        sliceColumn = matrixtest(:,checkLINE);
        founded = find(sliceColumn);
        P = [founded(2) checkLINE];
        
        % TRACE THE BOUNDARY

        boundary = bwtraceboundary(matrixtest,[P(1),P(2)],'N');
        for i=1:length(boundary)
            matrixDraw(boundary(i,1),boundary(i,2))=true;
        end
        
        % CLOSING AND FILLING 

        se = strel('disk',10);
        matrixDrawCLOSED = imclose(matrixDraw,se);
        matrixDrawFILLED = imfill(matrixDrawCLOSED,"holes");
        
       if(ismember(I,indexing_visualize)) %%%%%%%%%%%%%%% v %%%%%%%%%%%%%%%

            subplot(length(indexing_visualize),sizeSubplot,indexDRAW+4)
            imshow(matrixtest)
            hold on
            plot(boundary(:,2), boundary(:,1),'LineWidth',2,'Marker', 'none')
            title(["Boundary selected for slice " num2str(index)])

            subplot(length(indexing_visualize),sizeSubplot,indexDRAW+5)
            imshow(matrixDrawFILLED)
             title("Final region")
            % subplot(3,3,(I/5)-17)

            subplot(length(indexing_visualize),sizeSubplot,indexDRAW+6)
            Im = imshow(uint8(squeeze(immagine(:,:,I))));
              Im.AlphaData = 0.8;
                           hold on
            plot(boundary(:,2), boundary(:,1),'LineWidth',2,'Marker', 'none')
             title("Final region on slice " +num2str(I))
            hold off
            
       end %%%%%%%%%%%%%%%%%%%%%%%%%%%%% A %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % % UNCOMMENT FOR THE PATH FILLED
        % imshow(matrixDrawFILLED)

        % % UNCOMMENT FOR THE PATH
        % hold on
        % plot(boundary(:,2), boundary(:,1),'LineWidth',2,'Marker', 'none')
        % title(num2str(index))
    end


    VOLUMEmask(:,:,I) = matrixDrawFILLED;

end

figure(general)
sgtitle("Preprocessing")

% MANUAL CORRECTIONS

slice25 = VOLUMEmask(:,:,15);
slice66 = VOLUMEmask(:,:,66);

correcting_indexing_66 = 52:68;
correcting_indexing_25 = 16:24;

VOLUMEmask (:,:,correcting_indexing_66) = repmat(slice66,[1,1,correcting_indexing_66(end)-correcting_indexing_66(1)+1]);
VOLUMEmask (:,:,correcting_indexing_25) = repmat(slice25,[1,1,correcting_indexing_25(end)-correcting_indexing_25(1)+1]);


% EROSION 

VOLUMEeroso = imerode(VOLUMEmask,[1 0 1; 1 1 1; 1 0 1]);
VOLUMEeroso = imerode(VOLUMEeroso,[1 0 1; 1 1 1; 1 0 1]);
VOLUMEeroso = imerode(VOLUMEeroso,[1 0 1; 1 1 1; 1 0 1]);

% DILATE 

e = strel("line",1,1);

VOLUMEeroso = imdilate(VOLUMEeroso,e);
VOLUMEeroso = imdilate(VOLUMEeroso,e);
VOLUMEeroso = imdilate(VOLUMEeroso,e);


%volSmooth = imgaussfilt3(double(VOLUMEmask), 0.1);
clc
VOLUME_FINALE = zeros(156,206,155);
for kl = 1:155
   VOLUME_FINALE(:,:,kl) = immagine(:,:,kl).*VOLUMEeroso(:,:,kl);
end


VOLUME_FINALE = imerode(VOLUME_FINALE,[1 0 1; 1 1 1; 1 0 1]);

% PARAMETRI PER IL SEGMENTER
background = 1;
whiteMatter = 219;
greyMatter = 131;
csf = 62;
max_iter = 500;
tol_abs = 0.05;
slice = 100;

[sfondo, bianca, grigia,liquor,tutto] = Segmenter(VOLUME_FINALE,background,whiteMatter,greyMatter,csf,max_iter,tol_abs,slice);

%% %%%%%%%%%%%%%%%%%%%%%%% SALVATAGGIO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nii = make_nii(sfondo);
nii.hdr = T1.hdr;
save_nii(nii,'cluster_sfondo.nii');

nii = make_nii(bianca);
nii.hdr = T1.hdr;
save_nii(nii,'cluster_bianca.nii');

nii = make_nii(grigia);
nii.hdr = T1.hdr;
save_nii(nii,'cluster_grigia.nii');

nii = make_nii(liquor);
nii.hdr = T1.hdr;
save_nii(nii,'cluster_liquor.nii');

nii = make_nii(tutto);
nii.hdr = T1.hdr;
save_nii(nii,'cluster_all.nii');

%% %%%%%%%%%%%%%%%%%%%%%%% VOLUME VIEWER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%% COMPLETE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

viewer = viewer3d(BackgroundColor="white", ...
    GradientColor=[0.5 0.5 0.5],Lighting="on");
mriVol = volshow(tutto,Parent=viewer);

%% %%%%%%%%%%%%%%%%%%%%%%% SIMPLE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

volumeViewer(tutto)

%% %%%%%%%%%%%%%%%%%%%%%%% MASK BEFORE AND AFTER %%%%%%%%%%%%%%%%%%%%%%%%%%

viewer = viewer3d(BackgroundColor="white", ...
    GradientColor=[0.5 0.5 0.5],Lighting="on");
mriVol = volshow(VOLUMEeroso,Parent=viewer);

viewer = viewer3d(BackgroundColor="white", ...
    GradientColor=[0.5 0.5 0.5],Lighting="on");
mriVol = volshow(VOLUME_FINALE,Parent=viewer);


viewer = viewer3d(BackgroundColor="white", ...
    GradientColor=[0.5 0.5 0.5],Lighting="on");
mriVol = volshow(immagine,Parent=viewer);