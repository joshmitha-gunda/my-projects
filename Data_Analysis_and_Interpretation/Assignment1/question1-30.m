x=-3:0.02:3;
y=6.5*sin(2.1*x+pi/3);
z=y;
indices=randperm(301,90);
p=100+20*rand(1,90);

for k=1:90
    z(indices(k))=z(indices(k))+p(k);
end

y_median=zeros(1,301);
for k=1:301
    if k<=8
        N=z(1:k+8);
    elseif k<=292
        N=z(k-8:k+8);
    else
        N=z(k-8:301);
    end
    y_median(k)=median(N);
end

y_mean=zeros(1,301);
for k=1:301
    if k<=8
        N=z(1:k+8);
    elseif k<=292
        N=z(k-8:k+8);
    else
        N=z(k-8:301);
    end
    y_mean(k) =mean(N);
end

y_quartile =zeros(1,301);
for k = 1:301
    if k<=8
        N=z(1:k+8);
    elseif k<=292
        N=z(k-8:k+8);
    else
        N=z(k-8:301);
    end
    y_quartile(k) =quantile(N, 0.25); 
end

rel_mse_median =sum((y-y_median).^2)/sum(y.^2);
rel_mse_mean=sum((y - y_mean).^2)/sum(y.^2);
rel_mse_quartile =sum((y - y_quartile).^2)/sum(y.^2);

fprintf('median rel mse:%f\n',rel_mse_median);
fprintf('mean rel mse: %f\n',rel_mse_mean);
fprintf('quartile rel mse: %f\n',rel_mse_quartile);

figure;
plot(x,y,'w-','LineWidth',2); hold on;
plot(x,z,'b-','LineWidth',1);
plot(x,y_median,'r-','LineWidth',1.5);
plot(x,y_mean,'g-','LineWidth',1.5);
plot(x,y_quartile,'m-','LineWidth',1.5);
legend('Original','Corrupted','Median Filtered','Mean Filtered','Quartile Filtered','Location','best');

title('Moving filters comparison -f= 30%');
xlabel('x');
ylabel('Amplitude');