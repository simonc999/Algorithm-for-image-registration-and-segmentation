function [F] = imgReader(titleSG,inputImg1,inputImg2,inputImg3,inputImg4)
if nargin==5
    imgA = imread(inputImg1);
    imgB = imread(inputImg2);
    imgC = imread(inputImg3);
    imgD = imread(inputImg4);


    F=figure('Name',['Output immagini '  inputImg1]);
    sgtitle(titleSG, 'Interpreter', 'none')
    subplot(2,2,1)
    imshow(imgA)
    title({inputImg1 ' (A: immagine di riferimento)'}, 'Interpreter', 'none')
    subplot(2,2,2)
    imshow(imgB)
    title({inputImg2 ' (B: da rototraslare)'}, 'Interpreter', 'none')
    subplot(2,2,3)
    imshow(imgC)
    title({inputImg3 ' (C: doppio fattore di scala)'}, 'Interpreter', 'none')
    subplot(2,2,4)
    imshow(imgD)
    title({inputImg4 ' (D: singolo fattore di scala)'}, 'Interpreter', 'none')
end
if nargin==3
    if(ischar(inputImg1))
    imgA = imread(inputImg1);
    imgB = imread(inputImg2);
   
    else
        imgA = inputImg1;
        imgB = inputImg2;
        A = inputname(1);
         inputImg2 = inputname(3);
         inputImg1 = inputname(2);
    end
    
    
    [p_cong, ~, ~] = bihist(imgA, imgB);
    % cong_before = rot90(cong_before);
    F=figure('Name',['Analisi congiunta e differenza immagini '  inputImg1 ' e ' inputImg2]);
    sgtitle(titleSG, 'Interpreter', 'none')
    subplot(2,2,1)
    imshow(imgA)
    title(inputImg1, 'Interpreter', 'none')
    subplot(2,2,2)
    imshow(imgB)
    title(inputImg2, 'Interpreter', 'none')
    subplot(2,2,3)
    imshow(imgA-imgB);
    title([ inputImg1 ' - ' inputImg2], 'Interpreter', 'none')
    subplot(2,2,4)
    if strcmp(inputImg1,inputImg2)
        p_cong(p_cong~=0)=1;
    end
    imshow(imadjust(p_cong))
    title({'Istogramma congiunto di ' inputImg1 inputImg2}, 'Interpreter', 'none')

end

