% hopfiled数字识别
clc;clear;

%% 获取训练集
% 获取训练集图像
id = 1;
im = imread(['data/',num2str(id),'.jpg']); % 读取数字图像
im = double( imbinarize(im)); % 转为二值
start_im = im;
train_im = [im]';
% imshow(logical(train_im));

% 转为目标向量
[n, m] = size(train_im);
train_data = ones(n,m);
for i=1:n
    for j=1:m
        if train_im(i,j)==0
            train_data(i,j)=-1;
        end
    end
end

%% 建立hopfiled网络与测试
% 建立网络
net = newhop( train_data);

% 读取测试集图像
test_im = imread(['data/',num2str(id),'.jpg']); % 读取数字图像
[n, m] = size(test_im);
% 加入噪声
noisyNum = 100;
for i=1:noisyNum
    test_im(unidrnd(n), unidrnd(m)) = 255;
    %     test_im(unidrnd(n), unidrnd(m)) = 0;
end
show_test = test_im;

% 转为测试向量
test_im = double( imbinarize( test_im));
test_data = ones(n,m);
for i=1:n
    for j=1:m
        if test_im(i,j)==0
            test_data(i,j)=-1;
        end
    end
end

% 测试
test_out = sim(net, {n m}, {}, {(test_data)'});
test_out = test_out{n}';
% 转为可读
for i=1:n
    for j=1:m
        if test_out(i,j)==-1
            test_out(i,j)=0;
        end
    end
end

%% 显示
subplot(1,3,1);
imshow(start_im);
title('训练图');

subplot(1,3,2);
imshow(show_test);
title('测试图');

subplot(1,3,3);
imshow(test_out);
title('网络输出');
