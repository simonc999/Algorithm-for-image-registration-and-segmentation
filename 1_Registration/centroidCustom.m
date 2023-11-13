function CENTROID = centroidCustom(X,Y,W)

 switch nargin
        case 3
            
            W=W.^2;
            Y_centroid = sum(Y.*W)/sum(W);

        case 2
            
            W = ones(length(X(:,1)),1);           
            Y_centroid = sum(Y.*W)/sum(W);

     otherwise

            W = ones(length(X(:,1)),1);
            Y_centroid = 0;
            
 end
 
 X_centroid = sum(X.*W)/sum(W);

 CENTROID = [X_centroid Y_centroid];

end

