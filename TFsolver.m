function [result] = TFsolver(X, gt)

numOfClasses = size(unique(gt),1);
numOfViews = length(X);
numOfSamples = size(X{1}, 2);
MAXiter = 1000; % Maximum number of iterations for KMeans 
REPlic = 10; % Number of replications for KMeans

for i=1:numOfViews
    X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1);  %normalized
end

%Initialize Fs
for i=1:numOfViews
    [Fs{i},L{i}] = solveF(X{i}, numOfClasses);
    J{i} = zeros(numOfSamples,numOfClasses); 
    Q{i} = zeros(numOfSamples,numOfClasses); 
end
alpha = 1;
pho = 0.05;
mu = 2;
lambda = 1;
NITER = 20;
sX = [numOfSamples, numOfClasses, numOfViews]; 
omega = [23,66.5,57];

for iter = 1:NITER
    
    % == update Q{i} ==
    for i=1:numOfViews
        Q{i} = Q{i}+pho*(Fs{i}-J{i});
    end
    
    % == update J{i} ==
    F_tensor = cat(3, Fs{:,:});
    Q_tensor = cat(3, Q{:,:});
    f = F_tensor(:);
    q = Q_tensor(:);
    [j, objV] = wshrinkObj(f+1/pho*q,1/pho,sX,0,3,omega);
    J_tensor = reshape(j, sX); 
    for i=1:numOfViews
        J{i} = J_tensor(:,:,i);
    end
    
    % == update Fs{i} ==
    for i=1:numOfViews
        B{i} = lambda*eye(numOfSamples)-alpha*L{i};
        C{i} = pho/2*(J{i}-Q{i}/pho);
        [u,s,v]=svd(B{i}*Fs{i}+C{i},'econ');
        Fs{i}=u*v';   
    end
    
    % == update pho ==
    pho = pho*mu;
end

[Y, p, obj] = AWP(Fs);  %Step 2

[ACC,NMI,PUR] = ClusteringMeasure(gt,Y); 
[Fscore,Precision,R] = compute_f(gt,Y);
[AR,~,~,~]=RandIndex(gt,Y);
result = [ACC NMI PUR Fscore Precision R AR];

