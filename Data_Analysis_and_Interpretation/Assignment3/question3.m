values =dlmread('XYZ.txt');
X=values(:,1);
Y=values(:,2);
Z=values(:,3);
N=length(X);

P_matrix=[sum(X.^2),sum(X.*Y),sum(X);
    sum(X.*Y),sum(Y.^2),sum(Y);
    sum(X),sum(Y),N]; 
k_vector=[sum(X.*Z);sum(Y.*Z);sum(Z)];
parameters=inv(P_matrix)*(k_vector);
a=parameters(1);
b=parameters(2);
c=parameters(3);
Z_estimated=a*X+b*Y+c;
error=Z-Z_estimated;  
noise_variance=sum(error.^2)/(N);
fprintf('Estimated plane equation is: z=%.6f*x+%.6f*y+%.6f\n',a,b,c);
fprintf('Estimated noise variance is: %.6f\n',noise_variance);