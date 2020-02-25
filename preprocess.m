%train_data=csvread('train.csv',1,1);
x=xlsread('train.xlsx');
x=x(2:101,1:550);

s=size(x);
x=log(x+1);

maxs=zeros(s(1),1);
isnan=ismissing(x);
for i=1:s(1)
    maxx=0;
    for j=1:s(2)
        if isnan(i,j)==0 && x(i,j)>maxx
            maxx=x(i,j);
        end
    end
            
    maxs(i)=maxx;
end


for i=1:s(1)
    for j=1:s(2)
        if isnan(i,j)==1
            if j>1 && j<s(2)
                x(i,j)=(x(i,j-1)+x(i,j+1))/2;
            elseif j==1
                x(i,j)=x(i,2);
            else
                x(i,j)=x(i,j-1);
            end
        end
    end
end

for i=1:s(1)
    isnan_i=ismissing(x(i,:));
    xi=x(i,:);
    mu=mean(xi(~isnan_i));
    for j=1:s(2)
        if isnan_i(j)==1
            x(i,j)=mu;
        end
    end
end

pre=zeros(s(1),s(2)+1);

for i = 1:s(1)
    for j=1:s(2)
        pre(i,j)=x(i,j)/maxs(i);
    end
end

for i=1:s(1)
    pre(i,s(2)+1)=maxs(i);
end






