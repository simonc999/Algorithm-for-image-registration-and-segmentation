function [p_cong, p_marg1, p_marg2]=bihist(img1,img2)

lvl_gray=256;
% img1=imread(img1);
% img2=imread(img2);
dim= size(img2);
h=zeros(lvl_gray,lvl_gray);

for i=1:dim(1)
    for j=1:dim(2)
        c= img1(i,j)+1;
        r= img2(i,j)+1;
        h(r,c)= h(r,c)+1;
    end
end
h(1,1)=0; % ELIMINANO I LIVELLI CONGIUNTI E MARGINALI DI NERO
h(1,:)=0;
h(:,1)=0;
%imshow(imadjust(h));
p_cong= h./sum(sum(h)); 

p_marg1= sum(p_cong,1);  % le marginali sono i due istogrammi
p_marg2= sum(p_cong,2);

p_cong = rot90(p_cong);
end

