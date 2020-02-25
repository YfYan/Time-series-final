train_ = xlsread('preprocessed.xlsx');
test=xlsread('test.xlsx');
test=test(2:101,:);
train = train_(:,1:550);
window=[6,12,18,30];
s=0;
c=0;
submisson=[];
for i=1:100
    x=1:550;
    data_x=train(i,:);
    testi=test(i,:);
    m=train_(i,551);
    f=fit(x',data_x','fourier6');
    K=f(1:550);
    K=[K;zeros(60,1)];
    %data_x=[data_x,zeros(1,60)];
    pred=zeros(1,60);
    for j =1:60
            local=[];
            for l=1:4
                local=[local,mean(K(550+j-7:-7:550+j-window(l)*7-7))];
            end
            pred(j)=mean(local);
            K(550+j)=pred(j);
            data_x(550+j)=pred(j);
    end

    pred=pred*m;
    pred=exp(pred)-1;
    pred=ceil(pred);
    submisson=[submisson;pred];
    sss=smape(pred,testi);
   
    s=s+sss;
end

s=s/100




