%% ============================================================
%  Leave-One-Stress-Level-Out Validation for BPNN
%  TP321H at 550°C, stress levels 160-200 MPa
%  ============================================================

clear; clc; close all;

%% 1. 加载数据
% 训练数据：160,170,190,200 MPa（三列：应力，时间，应变）
train_file = 'C:\Users\13269\Desktop\何月\项目\321H\论文第二篇\321神经网络拟合多输入.xlsx';
raw_train = readmatrix(train_file);

% 180 MPa 数据：时间，应变
test_file = 'C:\Users\13269\Desktop\何月\项目\321H\论文第二篇\550度180MPa论文数据汇总\滤波后原始数据180MPa.xlsx';
raw_180 = readmatrix(test_file);
time_180 = raw_180(:,1);
strain_180 = raw_180(:,2);
[time_180, sortIdx] = sort(time_180);
strain_180 = strain_180(sortIdx);
stress_180 = 180 * ones(size(time_180));

% 合并所有五个应力的数据
% 训练数据中的四个应力
stress_levels = [160, 170, 190, 200];
data_all = cell(5,1);  % 存储五个应力的数据
for i = 1:4
    s = stress_levels(i);
    idx = raw_train(:,1) == s;
    time = raw_train(idx,2);
    strain = raw_train(idx,3);
    [time, sortIdx] = sort(time);
    strain = strain(sortIdx);
    data_all{i} = struct('stress', s*ones(size(time)), 'time', time, 'strain', strain);
end
% 第五个为180 MPa
data_all{5} = struct('stress', stress_180, 'time', time_180, 'strain', strain_180);

% 应力列表（顺序对应data_all）
stress_list = [160, 170, 190, 200, 180];
% 重新排序为递增：160,170,180,190,200
[stress_sorted, order] = sort(stress_list);
data_sorted = data_all(order);
stress_list = stress_sorted;

%% 2. 参数设置
best_arch = [2 2 4];   % 根据之前分组交叉验证选定的架构
num_stress = length(stress_list);

%% 3. 留一应力水平交叉验证
results = struct();
for i = 1:num_stress
    s_test = stress_list(i);
    fprintf('\n留出应力: %d MPa\n', s_test);
    
    % 训练集：除第i个外的所有应力
    train_idx = setdiff(1:num_stress, i);
    X_train = []; Y_train = [];
    for j = train_idx
        X_train = [X_train; data_sorted{j}.stress, data_sorted{j}.time];
        Y_train = [Y_train; data_sorted{j}.strain];
    end
    
    % 测试集：第i个应力
    X_test = [data_sorted{i}.stress, data_sorted{i}.time];
    Y_test = data_sorted{i}.strain;
    
    % 归一化（仅用训练集参数）
    [X_train_norm, ps_X] = mapminmax(X_train', 0, 1);
    [Y_train_norm, ps_Y] = mapminmax(Y_train', 0, 1);
    X_test_norm = mapminmax('apply', X_test', ps_X);
    
    % 创建并训练网络
    net = feedforwardnet(best_arch, 'trainlm');
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-6;
    net.trainParam.showWindow = false;
    for j = 1:length(best_arch)
        net.layers{j}.transferFcn = 'tansig';
    end
    net.layers{end}.transferFcn = 'purelin';
    net.divideFcn = 'dividetrain';
    [net, tr] = train(net, X_train_norm, Y_train_norm);
    
    % 预测
    Y_pred_norm = net(X_test_norm);
    Y_pred = mapminmax('reverse', Y_pred_norm, ps_Y);
    Y_pred = Y_pred';
    
    % 计算指标
    R2 = 1 - sum((Y_test - Y_pred).^2) / sum((Y_test - mean(Y_test)).^2);
    MAPE = mean(abs((Y_test - Y_pred) ./ Y_test)) * 100;
    RMSE = sqrt(mean((Y_test - Y_pred).^2));
    
    fprintf('  R2 = %.4f, MAPE = %.2f%%, RMSE = %.4f\n', R2, MAPE, RMSE);
    
    % 存储结果
    results(i).stress = s_test;
    results(i).R2 = R2;
    results(i).MAPE = MAPE;
    results(i).RMSE = RMSE;
    results(i).Y_test = Y_test;
    results(i).Y_pred = Y_pred;
    results(i).time = data_sorted{i}.time;
end

%% 4. 汇总结果
fprintf('\n===== 留一应力水平交叉验证汇总 =====\n');
fprintf('应力(MPa)\t类型\t\tR2\t\tMAPE(%%)\t\tRMSE\n');
for i = 1:num_stress
    s = results(i).stress;
    if s == 160 || s == 200
        type = '边界(外推)';
    else
        type = '内部(插值)';
    end
    fprintf('%d\t\t%s\t\t%.4f\t\t%.2f\t\t%.4f\n', ...
        s, type, results(i).R2, results(i).MAPE, results(i).RMSE);
end

% 计算内部应力平均指标
internal_idx = find(stress_list == 170 | stress_list == 180 | stress_list == 190);
mean_R2_internal = mean([results(internal_idx).R2]);
mean_MAPE_internal = mean([results(internal_idx).MAPE]);
mean_RMSE_internal = mean([results(internal_idx).RMSE]);
fprintf('\n内部应力(170,180,190 MPa)平均: R2=%.4f, MAPE=%.2f%%, RMSE=%.4f\n', ...
    mean_R2_internal, mean_MAPE_internal, mean_RMSE_internal);

%% 5. 绘制预测曲线对比图（可选）
figure('Position',[100 100 1200 800]);
for i = 1:num_stress
    subplot(2,3,i);
    plot(results(i).time, results(i).Y_test, 'b-', 'LineWidth', 1.5); hold on;
    plot(results(i).time, results(i).Y_pred, 'r--', 'LineWidth', 1.5);
    xlabel('Time (h)');
    ylabel('Creep strain (%)');
    title(sprintf('%d MPa (R^2=%.3f)', results(i).stress, results(i).R2));
    legend('Experimental', 'BPNN prediction');
    grid on;
end