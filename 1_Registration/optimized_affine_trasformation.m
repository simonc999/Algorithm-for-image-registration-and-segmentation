function [img_rot]=optimized_affine_trasformation(float_img,p)
% affine trasformation of 2D image with 6 dof( 1rotation, 2traslation,3scaling/shear )

% Passo l'immagine da ruotare (float_img) e il vettore p che contiene i valori relativi a rotazione e fattori di scala


if nargin == 0

    % default image (CASO STANDARD)

    float_img= imread('mri2_rot.gif');
    
    float_img= float_img(:,:,1);

    % default_parameters

    alf = deg2rad(-30);

    %----VETTORE TRASLAZIONE 
    % FORMA T= [tx; ty] DOVE tx E' LA TRASLAZIONE LUNGO X E ty TRASLAZIONE LUNGO Y
    % LA DEFAULT GIUSTAMENTE CORRISPONDE A TRASLAZIONE NULLA

    t= [0;0];

    %----MATRICE DI SCALA 
    % FORMA s= [sx 0; 0 sy] DOVE sx E' LA SCALA LUNGO X E sy SCALA LUNGO Y
    % LA DEFAULT GIUSTAMENTE CORRISPONDE A SCALA NULLA, MOLTIPLICANDO
    % ENTRAMBE LE DIMENSIONI PER 1

    s= [1 0;
        0 1];


else
    alf=p(1);
    t= [p(2);p(3)];
    s= [p(4), p(6); p(6),  p(5)];
end

% tic
% inizializzazione parametri

dim = size(float_img); 

img_rot = uint8(zeros(dim(1),dim(2)));

% INSERISCO LA MATRICE R ROTAZIONE IN 2D 
% !!! ATTENZIONE: alf POSITIVO IN SENSO ORARIO E NEGATIVO ANTIORARIO

rotation_matrix = [cos(alf)  -sin(alf); sin(alf) cos(alf)];

% creazione sisteme di coordinate traslato nell'origine
% DATE LE DIMENSIONI DELLA IMMAGINE CENTRIAMO NELL'ORIGINE
gap =0;

if dim(1)==256 && dim(2) == 256
    gap = 1;
end
x=-(floor(dim(2)/2)-gap):1:floor(dim(2)/2); 
y=-(floor(dim(1)/2)-gap):1:floor(dim(1)/2);

% sistema di coordinate originale

i=1:dim(1);
j=1:dim(2);

% vettori coodrinate(tutte le combinazionei x,y del piano)
% CREIAMO UNA MESHGRID SIA PER LE COORDINATE ORIGINALI i j SIA DELLE
% COORDINATE CENTRATE NELL'ORIGINE

[x,y] = meshgrid(x,y);
[j,i] = meshgrid(j,i);

x=x(:);y=y(:);
i=i(:);j=j(:);

% trasfomazione inversa dell'immagine

cord = (s*rotation_matrix)\[x';y'];
tr = (s*rotation_matrix)\t;

% cambio di sistema di rifermento nel sistema img

i_t = round(cord(2,:)-tr(2))+floor(dim(1)/2)+1 ;
j_t = round(cord(1,:)-tr(1))+floor(dim(2)/2)+1;
%controllo ed eliminazione valori esterni dal piano img
i_t1=i_t;j_t1=j_t;
m=[i_t1',j_t1',i,j];
m(i_t<=0|i_t>dim(1)|j_t<=0|j_t>dim(2),:)=[];
%j_t1(i_t<=0|i_t>dim(1)|j_t<=0|j_t>dim(2))=[];
%i(i_t<=0|i_t>dim(1)|j_t<=0|j_t>dim(2))=[];
%j(i_t<=0|i_t>dim(1)|j_t<=0|j_t>dim(2))=[];
%trasposiozione indice 
ind_t=sub2ind(dim,m(:,1),m(:,2));
ind=sub2ind(dim,m(:,3),m(:,4));
%scandisco tutta l'immagine trasformato e attraverso la trasformazione
%inversa posso recuperare i pixel precedenti alla trasformazione
%nell'immagine orginale
img_rot(ind)= float_img(ind_t);
%toc
%imshow(img_rot);
