n=1000;
mu=0;
sig=4;
samples=mu+sig*randn(n,1);
T=samples(1:750);
V=samples(751:1000);

sigma_vals=[0.001,0.1,0.2,0.9,1,2,3,5,10,20,100];

LL=zeros(11,1);

for i=1:11
    s=sigma_vals(i);
    loglik=0;
    
    for j=1:250
        sum_val=0;
        for k=1:750
            sum_val=sum_val+exp(-(V(j)-T(k))^2/(2*s^2));
        end
        p_est=sum_val/(750*s*sqrt(2*pi));
        loglik=loglik+log(p_est);
    end
    
    LL(i)=loglik;
end
[maxLL,idx]=max(LL);
best_sig=sigma_vals(idx);
fprintf('best sigma = %.3f\n',best_sig);

figure
plot(log(sigma_vals),LL,'-o')
xlabel('log sigma')
ylabel('log likelihood')
title('log likelihood vs log sigma')

x=-8:0.1:8;
p_kde=zeros(length(x),1);
for i=1:length(x)
    sum_val=0;
    for k=1:750
        sum_val=sum_val+exp(-(x(i)-T(k))^2/(2*best_sig^2));
    end
    p_kde(i)=sum_val/(750*best_sig*sqrt(2*pi));
end

p_true=(1/(sig*sqrt(2*pi)))*exp(-(x-mu).^2/(2*sig^2));

figure
plot(x,p_kde,'b')
hold on
plot(x,p_true,'r')
legend('KDE','true pdf')
title('KDE vs true density')

D=zeros(11,1);

for i=1:11
    s=sigma_vals(i);
    err=0;
    
    for j=1:250
        sum_val=0;
        for k=1:750
            sum_val=sum_val+exp(-(V(j)-T(k))^2/(2*s^2));
        end
        p_est=sum_val/(750*s*sqrt(2*pi));
        p_real=(1/(sig*sqrt(2*pi)))*exp(-(V(j)-mu)^2/(2*sig^2));
        
        err=err+(p_real-p_est)^2;
    end
    
    D(i)=err;
end

[minD,idx2]=min(D);
best_sig2=sigma_vals(idx2);
fprintf('sigma for min D = %.3f\n',best_sig2);
fprintf('D at best LL sigma = %.6f\n',D(idx));

figure
plot(log(sigma_vals),D,'-o')
xlabel('log sigma')
ylabel('D')
title('D vs log sigma')

p_kde2=zeros(length(x),1);
for i=1:length(x)
    sum_val=0;
    for k=1:750
        sum_val=sum_val+exp(-(x(i)-T(k))^2/(2*best_sig2^2));
    end
    p_kde2(i)=sum_val/(750*best_sig2*sqrt(2*pi));
end

figure
plot(x,p_kde2,'b')
hold on
plot(x,p_true,'r')
legend('KDE','true pdf')
title('KDE vs true density for min D')