function [F,LapN] = solveF(Z, numOfClasses)

numOfSamples = size(Z, 2);

%W = abs(Z')*abs(Z);   %Compute similarity matrix,(abs(Z{i})+abs(Z{i}'))/2
W = constructW_PKN(Z,10);
%W = Z'*Z;
DN=diag( 1./sqrt(sum(W, 2)+eps) );   %Diagonal matrix
LapN = speye(numOfSamples) - DN * W * DN;   %Normalized Laplacian matrix,LapN = diag(sum(W)) - W

%[V,D] = eig(double(LapN));   %Only compute double, notice that no single
%[D_sort, ind] = sort(diag(D));
%ind2 = find(D_sort>1e-6);
%F = V(:, ind2(1:numOfClasses));  %the first c eigenvector

[~,~,vN] = svd(LapN);
FN = vN(:,numOfSamples-numOfClasses+1:numOfSamples);
for i = 1:numOfSamples
    F(i,:) = FN(i,:) ./ norm(FN(i,:)+eps);
end
