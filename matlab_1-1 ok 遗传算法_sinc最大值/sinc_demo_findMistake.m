%%样例代码纠错
%%%错误处以注释+=========的形式给出

% 函数优化
% y = sin(x)/x -15<x<15
%

%主程序
clear all;
popsize=5; %群体大小
global chromlength; %字符串长度（个体长度）
global Cmin; %用于修正目标函数
chromlength=20;
Cmin=0;
pc=0.5; %交叉概率
pm=0.05; %变异概率
Max_Itet=100;
pop=initpop(popsize,chromlength); %随机产生初始群体

for i=1:Max_Itet %20 为迭代次数
    value=calobjvalue(pop); %计算目标函数
    fitvalue=calfitvalue(value); %计算群体中每个个体的适应度
    
    newpop=selection(pop,fitvalue); %复制
    newpop=crossover(newpop,pc); %交叉
    newpop=mutation(newpop,pc); %变异
    
    [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
    y(i)=max(bestfit);
    n(i)=i;
    pop3=bestindividual;
    x(i)=decodechrom(pop3,1,chromlength)*30/(2.^chromlength-1)-15;
    
    pop=newpop;
end

%最优的染色体
bestindividual
%最优解
x_best=decodechrom(bestindividual,1,chromlength)*30/(2.^chromlength-1)-15
%最优的目标值
bestfit

figure
fplot(@f_sinc, [-15, 15]) %函数句柄
hold on;
plot(x_best,bestfit,'r*');
hold off;

figure
its=1:Max_Itet;
plot(its,x,'g+:');
xlabel('Number of iteration')
ylabel('Best solution');

% figure
% plot(its,y,'g+:');
% xlabel('Number of iteration')
% ylabel('Best objective value');

%% 待优化目标函数
% y=sin(x)/x, -15<x<15
function [ output_args ] = f_sinc(x)
    % UNTITLED 此处显示有关此函数的摘要
    % 此处显示详细说明
    output_args=sinc(x);
end

%%
%初始化编码
function pop=initpop(popsize,chromlength)
    pop=round(rand(popsize,chromlength));
end

%%
%二进制数转化为十进制数
function pop2=decodechrom(pop,spoint,length)
    pop1=pop(:,spoint:spoint+length-1);
    pop2=decodebinary(pop1);
end
function pop2=decodebinary(pop)
    [px,py]=size(pop);
    %求 pop 行和列数
    for i=1:py
        pop1(:,i)=2.^(py-i).*pop(:,i);
    end
    pop2=sum(pop1,2);
    %求 pop1 的每行之和
end

%计算目标函数值
function value=calobjvalue(pop)
    global chromlength
    temp1=decodechrom(pop,1,chromlength); %将 pop 每行转化成十进制数
    x=temp1;
    x=temp1*30/(2.^chromlength-1)-15; %将二值域中的数转化为变量域的数
    %==============================================此处不-1？
    value=f_sinc(x); %计算目标函数值
    % value=x.^2;
end

%%
%计算个体的适应值
function fitvalue=calfitvalue(value)
    global Cmin;
    [px,py]=size(value);
    for i=1:px
        if Cmin+value(i)
            %===================================此处适应度,value为负数，适应度为负数
            %===================================因此+cmin将适应度改为正？cmin至少大于-1*函数最小值
            temp=Cmin+value(i);
        else
            temp=0;
        end
        fitvalue(i)=temp;
    end
    fitvalue=fitvalue';
end

%选择复制
function newpop=selection(pop,fitvalue)
    totalfit=sum(fitvalue); %求适应值之和
    fitvalue=fitvalue/totalfit; %单个个体被选择的概率
    fitvalue=cumsum(fitvalue); %如 fitvalue=[1,2,3,4],则 cumsum(fitvalue)=[1 3 6 10]
    
    [px,py]=size(pop);
    ms=sort(rand(px,1)); %从小到大排列
    fitin=1;
    newin=1;
    while newin<=px
        if(ms(newin))<fitvalue(fitin)
            %============================为什么要用累积概率作为选择指标
            %%===========================并没有按适应值排序
            %%===========================即累积概率大的，但适应度并不一定高
            %在不确bai定分析中，
            %当净现值期望du值相对值较低
            %需进一步了解发生在某一区间的可能性有多大，
            %则应计算这个区间内所有可能取值的概率之和，
            %即累积概率，用P(NPV≥0)表示。
            newpop(newin,:)=pop(fitin,:);
            newin=newin+1;
        else
            fitin=fitin+1;
        end
    end
end

%交叉
function newpop=crossover(pop,pc)
    [px,py]=size(pop);
    newpop=ones(size(pop)); %==========因为152行，因此最后一个 个体 变成了11111
    for i=1:2:px-1 % 一对一对的挑
        %===============================一对一对挑，群体为奇数，漏最后一个，然后150行
        if(rand<pc) % 发送交叉
            cpoint=round(rand*py);
            nwepop(i,:)=[pop(i,1:cpoint),pop(i+1,cpoint+1:py)]; %===========拼写错误
            newpop(i+1,:)=[pop(i+1,1:cpoint),pop(i,cpoint+1:py)];
        else
            newpop(i,:)=pop(i, :);
            newpop(i+1,:)=pop(i+1, :);
        end
    end
end

%变异
function newpop=mutation(pop,pm)
    [px,py]=size(pop);
    newpop=ones(size(pop));
    for i=1:px
        if(rand<pm)
            mpoint=round(rand*py);
            if mpoint<=0 %===============这样首位变异概率>其他位变异概率
                mpoint=1;
            end
            newpop(i,:)=pop(i,:);
            if newpop(i,mpoint)==0
                newpop(i,mpoint)=1;
            else
                newpop(i,mpoint)=0;
            end
        else
            newpop(i,:)=pop(i,:);
        end
    end
end

%求出群体中适应值最大的值
function [bestindividual,bestfit]=best(pop,fitvalue)
    [px,py]=size(pop);
    bestindividual=pop(1,:);
    bestfit=fitvalue(1);
    for i=2:px
        if fitvalue(i)>bestfit
            bestindividual=pop(i,:);
            bestfit=fitvalue(i);
        end
    end
end
