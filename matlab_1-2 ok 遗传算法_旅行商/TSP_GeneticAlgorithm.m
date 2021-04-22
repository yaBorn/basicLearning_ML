%% 遗传算法_旅行商问题
clear; clc;

%% 初始化数据
drawing = 0; % 迭代中绘制
probable_cross = 0.5; % 交叉概率
probable_mutation = 0.3; % 变异概率
%%% 参数赋值
popSize = 50; % 种群大小
iterNumMax = 5000; % 迭代次数

menNum = 5; % 旅行商数
menTourMin = 3; % 每个旅行商最少经历城市数

cityNum = 40; % 城市数
cityLocal = 10* rand( cityNum, 2); % 城市坐标[0,10]
aid = meshgrid(1:cityNum);
cityDis = reshape( sqrt( sum((cityLocal(aid,:)-cityLocal(aid',:)).^2, 2)) , cityNum, cityNum); % 城市距离矩阵

%%% 条件约束
menNum = max(1, min(cityNum,menNum)); % 1<=旅行商数<=城市数
menTourMin = max(1,min( floor(cityNum/menNum), menTourMin)); % 1<=个体最少经历数<=城市/商人数

%%% 种群初始化
popMens = zeros( popSize, menNum); % 每个商人走过的城市数
popRoad = zeros( popSize, cityNum); % 其依次走过的城市表
for i = 1:popSize
    popRoad(i, :) = randperm( cityNum); % 随机排表
    popMens(i, :) = func_randMen( menNum, menTourMin, cityNum); % 每个商人随机走n个城市
    %add(i)=sum(pop_brk(i,:));
end

%% 开始迭代
iterNew_popMens = zeros( popSize, menNum);
iterNew_popRoad = zeros( popSize, cityNum);
figdraw = figure;
for i=1:iterNumMax
    %%% 计算本次迭代个体旅途长度
    iterDis = func_getDis( popRoad, popMens, cityDis);
    [bestDis(i), bestID(i)] = min( iterDis); % 获得本次迭代最优个体
    
    %%% 判断迭代差分
    iterDiff = abs( diff(bestDis)); % 相邻迭代最短旅程差
    if i>110 % 迭代次数>25
        if sum( iterDiff( i-100:i-1))<1 % 最后100个绝对差和<1
            break;
        end
    end
    
    %%% 绘制最优个体
    if drawing==1
        func_drawBest( figdraw, bestID(i), popRoad, popMens, cityLocal, 0);
        title(['本次迭代最短路程: ',num2str(bestDis(i)), ' 迭代次数: ',num2str(i)]);
    end
    
    %%% 选择
    for j=1:popSize
        % 随机选择两个个体比较
        chA = randi( popSize);
        chB = randi( popSize);
        % 选择总路径短的繁衍
        if iterDis(chA)< iterDis(chB)
            iterCh = chA;
        else
            iterCh =chB;
        end
        iterNew_popMens( j,:) = popMens( iterCh,:);
        iterNew_popRoad( j,:) = popRoad( iterCh,:);
    end
    
    %%% 交叉
    for j=2:popSize
        % 遍历每个个体，随机与上个体交换popRoad
        if rand<probable_cross
            iterCross = iterNew_popRoad(j-1,:);
            iterNew_popRoad(j-1,:) =  iterNew_popRoad(j,:);
            iterNew_popRoad(j,:) = iterCross;
        end
    end
    
    %%% 变异
    for j=1:popSize
        % 遍历每个个体，随机交换同行的popRoad
        if rand<probable_mutation
            chA = randi( cityNum);
            chB = randi( cityNum);
            iterMutation = iterNew_popRoad(j, chA);
            iterNew_popRoad(j, chA) = iterNew_popRoad(j, chB);
            iterNew_popRoad(j, chB) = iterMutation;
        end
    end
    
    %%% 迭代
    popMens = iterNew_popMens;
    popRoad = iterNew_popRoad;
end

%%% 绘制最优个体
func_drawBest( figdraw, bestID(i), popRoad, popMens, cityLocal, 1);
title(['本次迭代最短路程: ',num2str(bestDis(i)), ' 迭代次数: ',num2str(i)]);

subplot(2,1,2);
plot(1:i, bestDis);
title(['迭代最短路程变化 ', ' 迭代次数: ',num2str(i)]);
axis([0 i 0 bestDis(1)]);

%% 函数
% 每个商人走过的城市数量的随机生成
function popMen = func_randMen( menNum, menTourMin, cityNum)
    % 1. 将cityNum个城市分为menNum个群
    % 2. 每个群内城市数>menTourMin
    % 3. 每个群即为某一个旅行商需要走的城市
    popMen = ones(1, menNum)* menTourMin;
    sub = cityNum- menTourMin*menNum; % 每个商人走至少数，剩余未走城市数
    for i=1:menNum-1
        add = fix( rand*sub); % 随机取未走的几个城市
        popMen(i) = popMen(i)+ add; % 加入i商人的旅途城市中
        sub = sub-add;
    end
    popMen(menNum) = popMen(menNum)+ sub;
    popMen = sort( popMen); % 此时popMen为每个商人走过的城市数
    popMen = cumsum(popMen); % 累加，此时popMen为该商人于popRoad中走过的最后列号
end

% 计算每次迭代种群个体中旅行商旅行总长度
function dis = func_getDis( popRoad, popMens, cityDis)
    % 种群个体数，商人数
    [popnum, mennum] = size(popMens);
    dis = zeros(popnum,1); % 个体所有商人旅行总长度
    
    % 计算每个个体旅程长度
    for i=1:popnum % 遍历个体
        lastk = 1;
        for j=1:mennum % 遍历个体中每个旅行商
            % 遍历该旅行商走过的城市
            for k=lastk:popMens(i,j)-1
                dis(i) = dis(i)+ cityDis( popRoad(i,k), popRoad(i,k+1));
            end
            % 该旅行商走过的首尾城市距离
            dis(i) = dis(i)+ cityDis( popRoad(i,lastk), popRoad(i,k+1));
            lastk = popMens(i,j)+1;
        end
    end
end

% 绘制最优个体情况
function func_drawBest( figdraw, id, popRoad, popMens, cityLocal, issubplot)
    figure(figdraw);
    if issubplot==1
        subplot(2,1,1);
    end
    
    [~, mennum] = size(popMens); % 商人数
    color = hsv(mennum);
    
    % 画城市点
    x = cityLocal(:,1);
    y = cityLocal(:,2);
    scatter(x,y, 'k*');
    hold on;
    
    % 绘制该个体中旅行商们旅行路线
    lastk = 1;
    for j=1:mennum % 遍历id个体中每个旅行商
        x1 = [];
        y1 = [];
        % 遍历该旅行商走过的城市并连接
        for k=lastk:popMens(id,j)-1
            x1 = [x1, cityLocal(popRoad(id,k),1), cityLocal(popRoad(id,k+1),1)];
            y1 = [y1, cityLocal(popRoad(id,k),2), cityLocal(popRoad(id,k+1),2)];
            %plot(x1, y1,'Color',color(j,:));
        end
        % 该旅行商走过的首尾连接
        x1 = [x1, cityLocal(popRoad(id,lastk),1), cityLocal(popRoad(id,k+1),1)];
        y1 = [y1, cityLocal(popRoad(id,lastk),2), cityLocal(popRoad(id,k+1),2)];
        plot(x1, y1,'Color',color(j,:));
        lastk = popMens(id,j)+1;
    end
    hold off;
end
