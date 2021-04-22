function varargout = M9_9(cityLocal,cityDis,menNum,menTourMin,popSize,iterNum,show_prog,show_res)
    % 输入：
    %     XY：各个城市坐标的N*2矩阵，N为城市的个数
    %     cityDis：各个城市之间的距离矩阵
    %     SALESMEN ：旅行商人数
    %     menTourMin：每个人所经过的最少城市点
    %     POP_SIZE：种群大小
    %     NUM_ITER：迭代次数
    % 输出：
    %     最优路线
    %     总距离
    
    % 初始化
    nargs = 8;
    for k = nargin:nargs-1
        switch k
            case 0
                cityLocal = 10*rand(40,2);
            case 1
                N = size(cityLocal,1);
                a = meshgrid(1:N);
                cityDis = reshape(sqrt(sum((cityLocal(a,:)-cityLocal(a',:)).^2,2)),N,N);
            case 2
                menNum = 5; %     SALESMEN：人数
            case 3
                menTourMin = 3; %     MIN_TOUR :每个人所经过的最少城市点
            case 4
                popSize = 80; % POP_SIZE：种群大小
            case 5
                iterNum = 5e3; % NUM_ITER：迭代次数
            case 6
                show_prog = 1;
            case 7
                show_res = 1;
            otherwise
        end
    end
    
    % 调整输入数据
    N = size(cityLocal,1);
    [nr,nc] = size(cityDis);
    if N ~= nr || N ~= nc
        error('Invalid XY or DMAT inputs!')
    end
    cityNum = N;
    
    % 约束条件
    menNum = max(1,min(cityNum,round(real(menNum(1)))));
    menTourMin = max(1,min(floor(cityNum/menNum),round(real(menTourMin(1)))));
    popSize = max(8,8*ceil(popSize(1)/8));
    iterNum = max(1,round(real(iterNum(1))));
    show_prog = logical(show_prog(1));
    show_res = logical(show_res(1));
    
    % 路线的初始化
    num_brks = menNum-1;
    dof = cityNum - menTourMin*menNum;
    addto = ones(1,dof+1);
    for k = 2:num_brks
        addto = cumsum(addto);
    end
    cum_prob = cumsum(addto)/sum(addto);
    
    % 种群初始化
    pop_rte = zeros(popSize,cityNum);
    pop_brk = zeros(popSize,num_brks);
    for k = 1:popSize
        pop_rte(k,:) = randperm(cityNum);
        pop_brk(k,:) = randbreaks();
    end
    tmp_pop_rte = zeros(8,cityNum);
    tmp_pop_brk = zeros(8,num_brks);
    new_pop_rte = zeros(popSize,cityNum);
    new_pop_brk = zeros(popSize,num_brks);
    
    % 图形中各个路线的颜色
    clr = [1 0 0; 0 0 1; 0.67 0 1; 0 1 0; 1 0.5 0];
    if menNum > 5
        clr = hsv(menNum);
    end
    
    % 遗传算法实现
    global_min = Inf;
    total_dist = zeros(1,popSize);
    dist_history = zeros(1,iterNum);
    if show_prog
        pfig = figure('Name','MTSP_GA | Current Best Solution','Numbertitle','off');
    end
    
    for iter = 1:iterNum
        for p = 1:popSize
            d = 0;
            p_rte = pop_rte(p,:);
            p_brk = pop_brk(p,:);
            rng = [[1 p_brk+1];[p_brk cityNum]]';
            for s = 1:menNum
                d = d + cityDis(p_rte(rng(s,2)),p_rte(rng(s,1)));
                for k = rng(s,1):rng(s,2)-1
                    d = d + cityDis(p_rte(k),p_rte(k+1));
                end
            end
            total_dist(p) = d;
        end
        
        % 种群中的最优路线
        [min_dist,index] = min(total_dist);
        dist_history(iter) = min_dist;
        if min_dist < global_min
            global_min = min_dist;
            opt_rte = pop_rte(index,:);
            opt_brk = pop_brk(index,:);
            rng = [[1 opt_brk+1];[opt_brk cityNum]]';
            if show_prog
                
                % 画最优路线图
                figure(pfig);
                for s = 1:menNum
                    rte = opt_rte([rng(s,1):rng(s,2) rng(s,1)]);
                    plot(cityLocal(rte,1),cityLocal(rte,2),'.-','Color',clr(s,:));
                    title(sprintf('Total Distance = %1.4f, Iterations = %d',min_dist,iter));
                    hold on
                end
                hold off
            end
        end
        
        %遗传算法的进程操作
        rand_grouping = randperm(popSize);
        for p = 8:8:popSize
            rtes = pop_rte(rand_grouping(p-7:p),:);
            brks = pop_brk(rand_grouping(p-7:p),:);
            dists = total_dist(rand_grouping(p-7:p));
            [ignore,idx] = min(dists);
            best_of_8_rte = rtes(idx,:);
            best_of_8_brk = brks(idx,:);
            rte_ins_pts = sort(ceil(cityNum*rand(1,2)));
            I = rte_ins_pts(1);
            J = rte_ins_pts(2);
            for k = 1:8 % 产生新的解
                tmp_pop_rte(k,:) = best_of_8_rte;
                tmp_pop_brk(k,:) = best_of_8_brk;
                switch k
                    case 2 % 选择交叉算子
                        tmp_pop_rte(k,I:J) = fliplr(tmp_pop_rte(k,I:J));
                    case 3 %变异算子
                        tmp_pop_rte(k,[I J]) = tmp_pop_rte(k,[J I]);
                    case 4 %重排序
                        tmp_pop_rte(k,I:J) = tmp_pop_rte(k,[I+1:J I]);
                    case 5 %修改城市点
                        tmp_pop_brk(k,:) = randbreaks();
                    case 6 %选择交叉
                        tmp_pop_rte(k,I:J) = fliplr(tmp_pop_rte(k,I:J));
                        tmp_pop_brk(k,:) = randbreaks();
                    case 7 % 变异，修改城市点
                        tmp_pop_rte(k,[I J]) = tmp_pop_rte(k,[J I]);
                        tmp_pop_brk(k,:) = randbreaks();
                    case 8 %重排序，修改城市点
                        tmp_pop_rte(k,I:J) = tmp_pop_rte(k,[I+1:J I]);
                        tmp_pop_brk(k,:) = randbreaks();
                    otherwise
                end
            end
            new_pop_rte(p-7:p,:) = tmp_pop_rte;
            new_pop_brk(p-7:p,:) = tmp_pop_brk;
        end
        pop_rte = new_pop_rte;
        pop_brk = new_pop_brk;
    end
    
    if show_res
        % 画图
        figure('Name','MTSP_GA | Results','Numbertitle','off');
        subplot(2,2,1);
        plot(cityLocal(:,1),cityLocal(:,2),'k.');
        title('City Locations');
        subplot(2,2,2);
        imagesc(cityDis(opt_rte,opt_rte));
        title('Distance Matrix');
        subplot(2,2,3);
        rng = [[1 opt_brk+1];[opt_brk cityNum]]';
        for s = 1:menNum
            rte = opt_rte([rng(s,1):rng(s,2) rng(s,1)]);
            plot(cityLocal(rte,1),cityLocal(rte,2),'.-','Color',clr(s,:));
            title(sprintf('Total Distance = %1.4f',min_dist));
            hold on;
        end
        subplot(2,2,4);
        plot(dist_history,'b','LineWidth',2);
        title('Best Solution History');
        set(gca,'XLim',[0 iterNum+1],'YLim',[0 1.1*max([1 dist_history])]);
    end
    
    % 结果展示
    if nargout
        varargout{1} = opt_rte;
        varargout{2} = opt_brk;
        varargout{3} = min_dist;
    end
    
    % 随机城市点的生成
    function breaks = randbreaks()
        if menTourMin == 1
            tmp_brks = randperm(cityNum-1);
            breaks = sort(tmp_brks(1:num_brks));
        else
            num_adjust = find(rand < cum_prob,1)-1;
            spaces = ceil(num_brks*rand(1,num_adjust));
            adjust = zeros(1,num_brks);
            for kk = 1:num_brks
                adjust(kk) = sum(spaces == kk);
            end
            breaks = menTourMin*(1:num_brks) + cumsum(adjust);
        end
    end
    
end
