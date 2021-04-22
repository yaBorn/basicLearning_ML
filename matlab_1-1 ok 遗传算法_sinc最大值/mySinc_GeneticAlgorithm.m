%% 函数优化
% y = sin(x)/x -15<x<15 最大值
%
clear all;

%% 主程序
% y = target_func( x); % 目标函数
global target_x; % 定义域(对称y轴)
global target_y; % 适应度参数 (求解定义域内=y的值) 或 (调整参数fmin)
target_y = 10;
target_x = 15;

co_num = 20; % 个体个数
co_length = 30; %个体长度

colony = initColony( co_num, co_length); % 初始化群体
colony_init = colony;
[ out_init_x, out_init_y] = countTargetFunc( colony_init);

probable_cross = 0.5; % 交叉概率
probable_mutation = 0.1; % 变异概率
MAX_lteration = 100; % 迭代次数

%%------遗传迭代
for i=1:MAX_lteration
    % 计算种群目标函数
    [ out_x, out_y] = countTargetFunc( colony);
    % 计算种群个体适应度
    fitness =  getFitness( out_y);
    
    % 记录变化前适应度与对应下标
    [ch_fitness(i), ch_i] = max( fitness);
    ch_fitY(i) = out_y(ch_i);
    
    % 选择
    new_colony = chSelection( colony, fitness);
    % 交叉
    new_colony = chCrossover( new_colony, probable_cross);
    % 变异
    new_colony = chMutation( new_colony, probable_mutation);
    
    colony = new_colony;
end
%%------记录最终适应度
[ch_fitness(i+1), ch_i] = max( fitness);
ch_fitY(i+1) = out_y(ch_i);

%%------显示
figr = 3;
figl = 2;
figure();
subplot(figr,figl,[1 2]);
fplot( @target_func, [-target_x, target_x]);
hold on;
plot( out_init_x, out_init_y, 'r*');
title('初始函数图像');
hold off;

subplot(figr,figl,[3 4]);
fplot( @target_func, [-target_x, target_x]);
hold on;
plot( out_x, out_y, 'r*');
title('最终函数图像');
hold off;

subplot(figr,figl,5);
x = 1:MAX_lteration+1;
plot(x,ch_fitness);
title('每轮最大适应值变化');

subplot(figr,figl,6);
x = 1:MAX_lteration+1;
plot(x,ch_fitY);
axis([0 inf -0.2 1.2]);
title( ['对应Y值变化',' 最终输出：',num2str( ch_fitY(i+1))] );

%% 目标函数
function [ out_y ] = target_func( x)
    out_y = sinc(x); % sinc 即 sin/x
end

%% 初始化群体
function out_colony = initColony( inNum, inLength)
    out_colony = round( rand( inNum, inLength));
    % num行，length列的随机小数
    % round四舍五入小数为01
end

%% 计算种群目标函数
function [ out_x, out_y] = countTargetFunc( colony)
    [n l] = size( colony);
    global target_x;
    out_x = getDecimalValue( colony); % 转为十进制
    out_x = out_x/(2.^l) *2*target_x -target_x; % 映射至变量域(-target_x,target_x)
    out_y = target_func( out_x);
end
%按行转为十进制
function out = getDecimalValue( colony)
    [n l] = size( colony);
    % 位数上的值转为十进制
    for i = 1:n % i个体
        for j = 1:l % j位
            colony2(i, j) = 2^(l-j) *colony(i, j);
        end
    end
    % 行相加
    out = sum( colony2, 2);
end

%% 计算种群个体适应度
function out = getFitness( y)
    global target_y;
    [n, ~] = size(y);
    for i=1:n
        % out(i) = 1/abs(target_y-y(i)); % 与目标y越接近，适应值越大 （完全一致？）
        out(i) = target_y + y(i); % 最大值 适应值即本身+足够大的target_y变为正数 最优情况-min
    end
    out = out'; % 转回列向量
end

%% 选择 复制 交叉 变异
%选择
function new_colony = chSelection( colony, fit_value)
    % 采用竞争选择
    [n,~] = size(colony);
    for i=1:n
        ch = randi([1 n],1,2); % 随机取两个个体
        if fit_value( ch(1))>fit_value( ch(2)) % 选择适应值大的个体
            new_colony(i, :) = colony(ch(1), :);
        else
            new_colony(i, :) = colony(ch(2), :);
        end
    end
end
%交叉
function new_colony = chCrossover( colony, probable_cross)
    [n l] = size(colony);
    new_colony = ones( [n l]);
    
    % 若为奇数个数，最后一位不交叉
    if mod(n,2) == 1
        new_colony(n, :) = colony(n, :);
    end
    
    % 开始交叉前2n位
    for i=1:2:n-1
        if rand<probable_cross % 发生交叉
            start = round( rand*l); % 第s位开始单点交叉
            new_colony(i, :) = [ colony(i, 1:start), colony(i+1, start+1:l)];
            new_colony(i+1, :) = [ colony(i+1, 1:start), colony(i, start+1:l)];
        else
            new_colony(i, :) = colony(i, :);
            new_colony(i+1, :) = colony(i+1, :);
        end
    end
end
% 变异
function new_colony = chMutation( colony, probable_mutation)
    [n l] = size( colony);
    new_colony = colony;
    for i=1:n
        if rand<probable_mutation % 发生变异
            start = randi(l); % 变异位置
            new_colony(i,start) = ~new_colony(i,start); %取反
        end
    end
end
