%% ID3决策树
% 决策类型：二值01
% 方案类型：二值01
% 数据格式：坏 否 低 无
%   更改格式，对预处理中trans2binary更改判断
clear;

% 数据预处理 二值化
in = readcell('data.xlsx');
[ matrix, label, active] = ID3_preprocess( in); % 数据 标签 属性历史

% 构造ID3决策树  左0右1
tree = ID3_creatTree( matrix, label, active);

% 测试决策树 
% outTest 最后一列     为根据决策树得到的该情况对应决策
% outTest 倒数第二列 为测试数据实际决策
% accuary为决策树准确率
[outTest, accuracy] = ID3_testTree( matrix, label, tree);

% 绘制决策树 左0右1
[IDs, values] = printTree_getID( tree);
figure();
printTree_plot( IDs, values);
title(['决策树输出准确率: ',num2str(accuracy)]);

%% 数据预处理
function [ matrix, label, active] = ID3_preprocess( in)
    % 输出
    % 标签label：记录属性和决策字符串
    % 数据矩阵matrix：存放数据对应属性与决策结果
    % 属性判断向量active：未判断该属性是，对应位置为1
    
    label = in(1, 2:end); % 首行标签 序号除外
    active = ones(1, length(label())-1); % 属性是否判断 1为未判断
    
    data = in(2:end, 2:end); % 数据矩阵
    [n, m] = size(data);
    matrix = zeros(n,m);
    for i=1:n
        for j=1:m
            matrix(i, j) = trans2binary( data(i, j)); % 将数据矩阵转为二值
        end
    end
end
% 将数据转为二值 识别：坏 否 低 无
function out = trans2binary( in)
    if strcmp(in, '坏') || strcmp(in, '否') ||...
            strcmp(in, '低') || strcmp(in, '无')
        out = 0;
    else
        out = 1;
    end
end

%% 构造ID3决策树 返回根节点  左0右1
function tree = ID3_creatTree( matrix, label, active)
    % 属性个数 数据个数
    numProperty = length(active);
    numMatrix = length(matrix(:,1));
    
    % 创建该树结点
    tree = struct('value','null', 'left','null', 'right','null');
    
    %%% 1 决策方案唯一，直接返回
    addAll = sum( matrix(:, numProperty+1)); % 数据矩阵最后一列为决策方案
    if addAll == numMatrix % 决策方案均成立
        tree.value = 'true';
        return;
    end
    if addAll == 0 % 决策方案均不成立
        tree.value = 'false';
        return;
    end
    
    %%% 2 属性均被判断，即遍历到底
    if sum( active) == 0
        if addAll >= numMatrix / 2 % 选择该情况中最可能的决策
            tree.value = 'true';
        else
            tree.value = 'false';
        end
        return
    end
    
    %%% 3 还有未判断属性，继续递归
    % 计算信息熵addEntropy
    % 决策1 -P[1]*log2P[1]
    p1 = addAll / numMatrix; % 概率
    if p1 == 0
        entropy1 = 0;
    else
        entropy1 = -1*p1*log2(p1); % 1的熵
    end
    % 决策0 -P[0]*log2P[0]
    p0 = (numMatrix - addAll) / numMatrix; % 概率
    if p0 == 0
        entropy0 = 0;
    else
        entropy0 = -1*p0*log2(p0); % 0的熵
    end
    % 信息熵H(S)
    addEntropy = entropy0+entropy1;
    
    % 计算各个属性期望熵 H(S|xi)
    % 期望熵H(S|xi)与最大熵addEntropy之差
    % 差越大，期望熵越小
    subEntropy = -1*ones( 1, numProperty);
    for i=1:numProperty
        if active(i) == 1 % 该属性还未判断 计算其期望熵 H(S|xi)
            % 统计该属性01条件数 和 对应的决策1数
            s1 = 0; s1_policy1 = 0;
            s0 = 0; s0_policy1 = 0;
            for j = 1:numMatrix % 遍历该属性
                if matrix(j, i)
                    s1 = s1+1;
                    if matrix(j, numProperty+1)
                        s1_policy1 = s1_policy1+1;
                    end
                else
                    s0 = s0 + 1;
                    if matrix(j, numProperty+1)
                        s1_policy1 = s1_policy1+1;
                    end
                end
            end
            
            % 计算 H(S1)
            % 计算该属性为1时，决策为1 / 0概率
            if ~s1 % 该属性无1
                p1 = 0;
                p0 = 0;
            else
                p1 = s1_policy1 / s1;
                p0 = (s1-s1_policy1) / s1;
            end
            % 根据概率计算 H(S1) (P[1]*log2P[1])
            if p1 == 0
                entropy1 = 0;
            else
                entropy1 = -1*p1*log2(p1);
            end
            % 根据概率计算 H(S1) (P[0]*log2P[0])
            if p0 == 0
                entropy0 = 0;
            else
                entropy0 = -1*p0*log2(p0);
            end
            %  H(S1)
            entropy_s1 = entropy0+entropy1;
            
            % 计算 H(S0)
            % 计算该属性为0时，决策为1 / 0概率
            if ~s0 % 该属性无0
                p1 = 0;
                p0 = 0;
            else
                p1 = s0_policy1 / s0;
                p0 = (s0-s0_policy1) / s0;
            end
            % 根据概率计算 H(S0) (P[1]*log2P[1])
            if p1 == 0
                entropy1 = 0;
            else
                entropy1 = -1*p1*log2(p1);
            end
            % 根据概率计算 H(S0) (P[0]*log2P[0])
            if p0 == 0
                entropy0 = 0;
            else
                entropy0 = -1*p0*log2(p0);
            end
            %  H(S0)
            entropy_s0 = entropy0+entropy1;
            
            % H(S0) 和 H(S1)带入条件熵公式
            hope = (s1/numMatrix)*entropy_s1+ (s0/numMatrix)*entropy_s0;
            subEntropy(i) = addEntropy-hope;
        end
    end
    
    % 选择最小期望熵（最大熵差）并扩展结点
    % 对应属性对该节点扩展
    [~, choose] = max( subEntropy); % 对应属性位置
    tree.value = label{choose}; % 结点赋值
    active(choose) = 0; % 该属性已判断
    
    % 根据该属性划分数据
    matrix0 = matrix( matrix(:,choose)==0, :); % 数据中该属性为0的部分
    matrix1 = matrix( matrix(:,choose)==1, :); % 数据中该属性为1的部分
    
    % 递归子树 左0 右1
    % 左子树
    if isempty( matrix0) % 当此数据为空
        if (addAll >= numMatrix / 2) % 选择该情况中最可能的决策
            leaf = struct('value', 'true', 'left', 'null', 'right', 'null');
        else
            leaf = struct('value', 'false', 'left', 'null', 'right', 'null');
        end
        tree.left = leaf;
    else % 不为空，递归
        tree.left = ID3_creatTree( matrix0, label, active);
    end
    % 右子树
    if isempty( matrix1) % 当此数据为空
        if (addAll >= numMatrix / 2) % 选择该情况中最可能的决策
            leaf = struct('value', 'true', 'left', 'null', 'right', 'null');
        else
            leaf = struct('value', 'false', 'left', 'null', 'right', 'null');
        end
        tree.left = leaf;
    else % 不为空，递归
        tree.right = ID3_creatTree( matrix1, label, active);
    end
    
    return
end

%% 打印ID3决策树
% 先序遍历树 返回顺序编号
function [ IDs, values] = printTree_getID( tree)
    % 输出：
    % 遍历顺序编号：nodeID
    % 对应结点值：nodeValue
    this_id = 0; % 当前结点编号
    IDs( 1) = 0; % 输出遍历结点编号（根为0）
    values = []; % 对应结点值
    
    % 先序遍历tree
    queue = [ [ ], tree]; % 根节点入队
    while ~isempty( queue) % 队列非空
        % 出队
        node = queue(1);
        queue = queue(2:end);
        
        % 更新编号表和数值表
        this_id = this_id +1; % 更新当前编号
        if ~isleaf(node) % 不是叶子
            % 子节点的父为当前结点
            % 赋值对应编号表
            IDs( this_id +length(queue)+1) = this_id;
            IDs( this_id +length(queue)+2) = this_id;
        end
        values{1,this_id} = node.value;
        
        
        % 子结点入队
        if ~strcmp(node.left,'null') % 左子树不为空
            queue = [queue, node.left]; % 进队
        end
        if ~strcmp(node.right,'null') % 左子树不为空
            queue = [queue, node.right]; % 进队
        end
    end
end
% 判断叶子节点
function out = isleaf( node)
    if strcmp(node.left,'null') && strcmp(node.right,'null') % 无子节点
        out =1;
    else
        out=0;
    end
end
% 根据队列绘制 改写m函数treeplot
function printTree_plot( IDs, values)
    [x,y,h] = treelayout( IDs);
    leaf_n = find( IDs~=0); % 非根结点下标
    leaf_id = IDs( leaf_n); % 费根结点编号值
    X = [x(leaf_n); x(leaf_id); NaN(size(leaf_n))];
    Y = [y(leaf_n); y(leaf_id); NaN(size(leaf_n))];
    
    X = X(:);
    Y = Y(:);
    n = length(IDs);
    if n < 500
        hold on ;
        plot (x, y, 'ro', X, Y, 'r-');
        nodesize = length(x);
        for i=1:nodesize
            text( x(i)+0.01, y(i), values{1,i}); % 标记标签
        end
        hold off;
    else
        plot (X, Y, 'r-');
    end
    xlabel(['height = ' int2str(h)]);
    axis([0 1 0 1]);
end

%% 测试ID3决策树
function [ outTest, accuracy,m] = ID3_testTree( matrix, label, tree)
    [n, m] = size( matrix);
    outTest = [matrix, zeros(n,1)]; % 增加一列根据树查询结果
    
    % 遍历测试数据
    for i=1:n
        this_node = tree; % 当前指针位于根节点
        
        % 指针按照测试例到叶子
        while ~isleaf(this_node)
            index = ismember( this_node.value, label); % 查找该结点值位于label的列号
            if matrix( i, index) == 0 % 该测试例子在此属性的选择
                this_node = this_node.left;
            else
                this_node = this_node.right;
            end
        end
        
        % 输出查询结果
        if this_node.value == "true"
            outTest( i, m+1) = 1;
        else
            outTest( i, m+1) = 0;
        end
    end
    
    % 计算准确率
    same = outTest(:, m) + outTest(:, m+1);
    % m列为真实结果，m+1为决策树结果
    % 两者相加，0 2为相同，1为两者不同
    accuracy = (n- sum(same==1))/ n;
end


