%% BP非线性分类器
clc;
clear;
figure
%% 数据准备
dataNum = 1000; % 点集数
learn_rate = 0.02;   % 学习率
train_times = 2000; % 训练次数
hidden_nodes = 15;   % 隐层节点数
radio = 0.3; % 分类半径

%%% 训练点集/标签 测试点集/标签
% 标签为首列1/0，二列取反
% BP网络输出标签取max为1，min为0
% max位置与标签1位置相同时，视为输出正确
[train_local, train_target, test_local, test_target] = func_getData( dataNum, radio);

%% 预定义权值
% 隐含层
W1 = rand(hidden_nodes,2);
b1 = zeros(hidden_nodes,1);
% 输出层
W2 = rand(2,hidden_nodes);
b2 = zeros(2,1);

%% 训练
out_mse = zeros(train_times,1); % 训练过程中的契合度
[trainNum,~] = size(train_target);
runCount = 0; % 训练次数

%%% 开始迭代
for j = 1:train_times
    sumMSE = 0; % 计算误差
    %%% 顺序输入样本
    for i = 1:trainNum % 顺序方式 每输入一个样本，调整权值
        % 前向传播
        target = train_target(i,:)'; % 标签
        X = train_local(i,:)'; % 坐标
        layer = func_sigmoid(W1*X +b1); % 隐含层输出
        out = func_sigmoid(W2*layer +b2); % 输出层输出
        
        % 反向传播
        % 链式法则计算损失到各个权值的偏导
        error = target - out; % 误差
        delta_out = -diag(out.*(1-out))*error;
        %delta_hide = diag(layer1.*(1-layer1))*W2'*delta_out;
        delta_hide = (delta_out(1)*W2(1,:)+delta_out(2)*W2(2,:))'.*layer.*(1-layer);
        
        % 调整隐含层与输出层权值
        update_W2 = delta_out *layer';
        update_W1 = delta_hide *X';
        W2 = W2 - update_W2 *learn_rate;
        W1 = W1 - update_W1 *learn_rate;
        sumMSE = sumMSE + sum(error .* error);
    end
    
    %%% 计算本次训练误差
    sumMSE = sqrt(sumMSE) / trainNum; % 每个样本输入完毕，计算本次均方误差
    out_mse(j) = sumMSE;
    fprintf("mse = %.4f\r\n",sumMSE);
    if sumMSE == 0.01
        break;
    end
    runCount = runCount + 1;
end

%% 测试
[testNum,~] = size(test_target);
out_test = zeros(testNum,2); % 输出结果
out_right = zeros(testNum,1); % 分类正确则为1
out_accuracy = 0; % 准确率
for i = 1:testNum
    target = test_target(i,:)'; % 标签
    X = test_local(i,:)'; % 坐标
    layer = func_sigmoid(W1*X+b1); % 隐含层输出
    out = func_sigmoid(W2*layer+b2); % 输出层输出
    out_test(i,:) = round(out); % 储存输出
    
    if find(target==1) == find(out==max(out)) % 判断权值与标签是否一致
        out_accuracy = out_accuracy+1;
        out_right(i) = 1;
    end
end

%% 打印
%%% 训练集图
subplot(1,3,1);
plot(train_local(train_target(:,1)==1,1), train_local(train_target(:,1)==1,2),'c.'); hold on;
plot(train_local(train_target(:,1)==0,1), train_local(train_target(:,1)==0,2),'g.'); hold off;
title('训练集');
%%% 训练集与测试结果图
subplot(1,3,2);
plot(train_local(train_target(:,1)==1,1), train_local(train_target(:,1)==1,2),'c.'); hold on;
plot(train_local(train_target(:,1)==0,1), train_local(train_target(:,1)==0,2),'g.');
plot(test_local(out_test(:,1)==1,1), test_local(out_test(:,1)==1,2),'bo');
plot(test_local(out_test(:,1)==0,1), test_local(out_test(:,1)==0,2),'ro');
plot(test_local(out_right(:)==0,1), test_local(out_right(:)==0,2),'k*');
hold off;
%%% 输出准确率 
out_accuracy = out_accuracy/testNum;
fprintf("accuracy : %.2f\r\n",out_accuracy);
title([' 测试 准确率:',num2str(out_accuracy)]);

%%% 训练中误差变化
subplot(1,3,3);
x_axis = 1:1:train_times;
plot(x_axis,out_mse); % 联结权值按误差梯度下降的方向迭代
title(['训练误差变化 最终误差:',num2str(out_mse(runCount))]); 

%% 获取数据集
function [train_local, train_target, test_local, test_target] = func_getData( dataNum, radio)
    %%% 生成数据
    % 随机生成数据
    local_data = rand(dataNum, 2); % 坐标
    target_data = zeros(dataNum, 1); % 分类
    % 给数据分类做标记
    for i = 1:dataNum
        if sqrt((local_data(i,1)-0.5)^2+(local_data(i,2)-0.5)^2) < radio
            % 距离在半径内的点为1
            target_data(i,1) = 1;
        else
            target_data(i,1) = 0;
        end
    end
    target_group = [target_data,1-target_data];
    
    %%% 选取90%为训练集，10%为测试集
    index = randperm(dataNum,round(dataNum*0.9));
    train_local = local_data(index,:);
    train_target = target_group(index,:);
    test_local = local_data;
    test_local(index,:) = [];
    test_target = target_group;
    test_target(index,:)=[];
end

%% 激发函数 使用P202的单极性S函数
function y = func_sigmoid(x)
    y = 1./(1+exp(-x));
end

