function [I_A, I_B, xg_A, xg_B, yg_A, yg_B, X_P, Y_P] = ParMethodFun(a,b)
Q = size(a);
ind_A = find(a > 0);
ind_B = find(b > 0);

[row_A, col_A] = ind2sub(Q,ind_A); % coordinate xy immagine A
[row_B, col_B] = ind2sub(Q,ind_B); % coordinate xy immagine B

X_P = 1:1:size(a,2);
Y_P = 1:1:size(a,1);

xg_A = sum(col_A)/length(col_A); % x centroide
yg_A = sum(row_A)/length(row_A); % y centroide

xg_B = sum(col_B)/length(ind_B);
yg_B = sum(row_B)/length(ind_B);

Ixx_A = 0;
Iyy_A = 0;
Ixy_A = 0;
Ixx_B = 0;
Iyy_B = 0;
Ixy_B = 0;
% Izz_A = 0;
% Izz_B = 0;

for i=1:size(a,2)
    for j=1:size(b,1)
        Ixx_A = Ixx_A + ((Y_P(j)-yg_A)^2)*a(j,i);
        Ixx_B = Ixx_B + ((Y_P(j)-yg_B)^2)*b(j,i);
        Iyy_A = Iyy_A + ((X_P(i)-xg_A)^2)*a(j,i);
        Iyy_B = Iyy_B + ((X_P(i)-xg_B)^2)*b(j,i);
        Ixy_A = Ixy_A + (((X_P(i)-xg_A)*(Y_P(j)-yg_A))*a(j,i));
        Ixy_B = Ixy_B + (((Y_P(j)-yg_B)*(X_P(i)-xg_B))*b(j,i));
%       Izz_A = Izz_A + ((x(i)-xg_A)^2 + (y(j)-yg_A)^2)*a(j,i);
%       Izz_B = Izz_B + ((x(i)-xg_B)^2 + (y(j)-yg_B)^2)*b(j,i);
    end
end

I_A = [Ixx_A, -Ixy_A; -Ixy_A, Iyy_A];
I_B = [Ixx_B, -Ixy_B; -Ixy_B, Iyy_B];