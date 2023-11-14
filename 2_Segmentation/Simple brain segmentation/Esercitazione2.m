% ESERCITAZIONE 2
% SEGMENTAZIONE CERVELLO

clear 
close all
clc


T1 = load_untouch_nii('S01_T1_atlas_space_brain.nii');
matrice = T1.img;


figure("Name","Slice viewer --> default selected:56")
title("default = 56")
sliceViewer(matrice);

% VISUALIZZO UNA SLICE

index_slice = 56;

% OSSERVO L'ISTOGRAMMA E TROVO I PICCHI
figure('Name','Istogramma dei livelli di grigio')
sgtitle(['Istogramma slice ' , num2str(index_slice)])
imhist(uint8(squeeze(matrice(:,:,index_slice))))
grid on;
xlim([0, 260]);
ylim([0, 100]);

% PARAMETRI PER IL SEGMENTER
background = 1;
whiteMatter = 219;
greyMatter = 131;
csf = 62;
max_iter = 500;
tol_abs = 0.05;
slice = 56;

[sfondo, bianca, grigia,liquor,tutto] = Segmenter(matrice,background,whiteMatter,greyMatter,csf,max_iter,tol_abs,slice);

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


