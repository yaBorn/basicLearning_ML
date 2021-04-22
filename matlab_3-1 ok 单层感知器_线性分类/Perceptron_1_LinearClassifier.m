%% 单层感知器 神经网络 用于线性分类任务
% 设置平面上两组散点，其线性可分
% 构建单层感知器，对其划分超平面
clear ;

%% 训练集准备
% 因为正态分布的随机性,有低概率出现线性不可分。
NUM = 50; % 元素个数/2

train_1 = 3 + randn(NUM, 2); % 第一组横纵坐标 正态分布
train_1 = [train_1, ones(NUM, 1)]; % 标签1（激活函数）
train_2 = 8 + randn(NUM, 2); % 第二组
train_2 = [train_2, -1*ones(NUM, 1)]; % 标签-1
 % 合并为一个训练数据集 
 % 其每列对应书中x1(t) x2(t) 激活函数
train = cat(1,train_1,train_2);

%% 感知器训练
% 超平面 w*x + b = 0
% 对于两组样本有 w*xi+b分别</> 0
learnRate = 0.4; % 学习率（增益因子(0,1]）
w = randi([-100, 100], 2, 1); % 权值向量 超平面w=w1/w2
b = randi([-100, 100]); % 阈值

times = 1; % 记录迭代信息
errNums(1) = NUM;
while true
    % 损失函数最优化求解，用梯度下降求解 
    % 计算网络输出 Σwi*xi + b :样本输入矩阵train(:, 1:2) 矩阵乘 权值向量w
    y = train(:, 3) .*( train(:, 1:2)*w +b); % 计算输出
    
    % 两组样本带入超平面分别有w*xi+b </> 0
    % 因此(Σwi*xi+b) *激活函数train(:, 3)使得稳定样本输出均>0
    errorID = find( y<=0); % 误分类点行号
    errorNum = length( errorID);
    times = times +1; % 记录迭代信息
    errNums(times) = errorNum;
    if errorNum == 0 % w b对于所有样本均稳定，算法结束
        break;
    end
    
    % 有不稳定样本(误分类点)
    id = errorID( randi([1, errorNum])); % 随机选择一个误分类点
    % 调整权值
    w = w + learnRate.* train(id, 3).* train(id, 1:2)';
    b = b + learnRate.* train(id, 3);
end

%% 绘制
%figure();
subplot(2,1,1);
% 两组点
plot(train_1(:, 1),train_1(:, 2),'b*',train_2(:, 1),train_2(:, 2),'r*');  
hold on;
% 超平面
lineX =  linspace(0, 12);
lineY = (-w(1)*lineX -b) /w(2); % 输出超平面
plot(lineX, lineY, 'c', 'linewidth', 1.5);
hold off;
axis([0,12,0,12]);
title('超平面划分');

errNums = errNums';
subplot(2,1,2);
plot(1:times, errNums);
title(['训练不稳定(误分)点数 迭代次数: ',num2str(times)]);

