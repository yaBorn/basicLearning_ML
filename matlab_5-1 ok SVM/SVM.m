%% SVM分类器
clc;clear;

% kertype = 'linear';  % 线性核
kertype = 'rbf'; % 高斯核
dataNum = 30; % 数据个数/2
C = 10;  % 成本约束参数
epsilon = 1e-8; % 支持向量选择范围

%% 数据准备
% 两组坐标
dataLocal1 = randn(2,dataNum); % 2行N列矩阵, 元素服从正态分布
dataLocal2 = 4+randn(2,dataNum); % 2*N矩阵, 元素服从正态分布且均值为5，测试高斯核可x2 = 3+randn(2,n);
% 对应标签
dataTarget1 = ones(1,dataNum); %1*N个1
dataTarget2 = -ones(1,dataNum); %1*N个-1
% 显示
plot(dataLocal1(1,:),dataLocal1(2,:),'bo',dataLocal2(1,:),dataLocal2(2,:),'ko');
axis([-3 8 -3 8]);

%% SVM
%%% 获得支持向量
X = [dataLocal1,dataLocal2]; % 坐标集
Y = [dataTarget1,dataTarget2]; % 标签集
outSVM = func_svmTrain(X, Y, kertype, C, epsilon); % 训练

%%% 超平面判别
% 初始化超平面
[hyperLocal1,hyperLocal2] = meshgrid(-3:0.05:8,-3:0.05:8);  % 初始化超平面采样矩阵，网格采样点从-3到8,间隔0.05
[rows,cols] = size(hyperLocal1);
nt = rows*cols;
Xt = [reshape(hyperLocal1,1,nt); reshape(hyperLocal2,1,nt)]; % 超平面初始点集
% 判别
outTest = func_svmTest(outSVM, Xt, kertype);

%% 显示输出
hold on;
% 标记支持向量
plot(outSVM.Xsv(1,:),outSVM.Xsv(2,:),'r*');
% 绘制超平面
Yd = reshape(outTest.Y,rows,cols); % 将点集判别结果重新排列为采样点矩阵
contour(hyperLocal1,hyperLocal2,Yd); % 绘制登高线的方式将超平面-1/1的界限绘制出来
title('svm分类结果图');
hold off;

%% 获取支持向量
function outSVM = func_svmTrain(X, Y, kertype, C, epsilon)
    %%% 输入
    % X：坐标集 2*Num 值为坐标分布
    % Y：标签集 1*Num 值为-1/1标签
    % kertype：核函数类型 (线性核/高斯核)
    % C：成本约束参数
    % epsilon：支持向量选择范围  1e-8 
    %%% 输出支持向量
    % a：支持向量对应在二次规划下的解
    % Xsv：支持向量坐标
    % Ysv：支持向量标签
    % Num：支持向量个数
    
    %%% 二次规划求解
    % 使用quadprog具有线性约束的二次目标函数的求解器
    dataNum = length(Y); % 获取数据个数
    H = (Y'*Y) .*func_kernel(X,X,kertype); % 根据核函数计算数据二次目标项, 为对称实矩阵
    f = -ones( dataNum,1); % 线性目标项
    % 以 1/2*x'*H*x+f'*x表达式形式表示二次矩阵
    A = []; b = []; % 不等式约束 空
    Aeq = Y; % 线性等式约束, 类比标签
    beq = 0; % 线性等式约束 空
    lb = zeros( dataNum,1); % 下界
    ub = C *ones( dataNum,1); % 上界
    a0 = zeros( dataNum,1);  % 解的初始近似值
    options = optimset; % options为quadprog算法优化选项，创建默认值
    options.LargeScale = 'off'; % 大规模搜索, off表示规模搜索模式关闭
    options.Display = 'off'; % 不显示输出
    % 二次目标函数的求解
    % a为满足所有边界和线性约束的条件下对 1/2*x'*H*x + f'*x 进行最小化的向量
    a  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
    
    %%% 获得支持向量
    sv_label = find(abs(a)>epsilon); % epsilon<a<a(max)则认为x为支持向量
    outSVM.a = a(sv_label);
    outSVM.Xsv = X(:,sv_label); % 支持向量坐标
    outSVM.Ysv = Y(sv_label);
    outSVM.Num = length(sv_label);
end

%% 超平面分类
function outTest = func_svmTest(outSVM, Xt, kertype)
    %%% 输入
    % svm：支持向量
    % Xt：超平面点集
    % kertype：核函数
    %%%  输出
    % outTest.Y：超平面点集判别结果
    % outTest.score：判别值
    
    temp = (outSVM.a'.*outSVM.Ysv) *func_kernel(outSVM.Xsv, outSVM.Xsv, kertype); % 支持向量权重
    b = mean( outSVM.Ysv -temp);  % 支持向量权重与实际标签的差均值作为阈值
    w = (outSVM.a'.*outSVM.Ysv) *func_kernel(outSVM.Xsv, Xt, kertype); % 超平面点集权重
    Y = sign(w +b);  % 超平面点集分类判别
    
    outTest.score = w +b; % 超平面判别值
    outTest.Y = Y; % 超平面判别结果
end

%% 核函数
function out = func_kernel(X, Y, kertype)
    %X 维数*个数
    switch kertype
        case 'linear'   %线性核
            out = X'*Y;
        case 'rbf'      %高斯核
            delta = 5; % 核大小
            delta = delta*delta;
            XX = sum(X'.*X',2); %2表示将矩阵中的按行为单位进行求和
            YY = sum(Y'.*Y',2);
            XY = X'*Y;
            out = abs( repmat(XX,[1 size(YY,1)]) +repmat(YY',[size(XX,1), 1])-2*XY);
            out = exp(-out./delta);
    end
end
