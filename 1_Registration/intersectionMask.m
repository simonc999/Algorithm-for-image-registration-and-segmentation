function [output] = intersectionMask(alfa,tx,ty,sx,sy,sxy,dim)
    output = double(optimized_affine_trasformation(uint8(ones(dim)),[alfa tx ty sx sy sxy]));
    output(output==0) =NaN; % RIMOZIONE DEI PIXEL FUORI DALL'AREA DI INTERSEZIONE
end

