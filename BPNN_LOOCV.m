%% ============================================================
%  Leave-One-Internal-Stress-Curve-Out Cross-Validation for BPNN
%  数据来源：data.xlsx（三列：应力、时间、应变）
%  架构选择仅使用内部应力 170 和 190 MPa
%  180 MPa 严格留出作为独立测试集
%  160 和 200 MPa 为边界应力（外推），不用于架构选择
%  ============================================================

clear; clc; close all;

%% 1. 加载全部数据
data_file = 'data.csv','Sheet1';
raw = readmatrix(data_file);

% 检查数据
fprintf('数据总行数: %d\n', size(raw,1));
stress_all = unique(raw(:,1));
fprintf('数据中包含的应力水平: %s\n', mat2str(stress_all'));

%% 2. 分离训练数据和测试数据
test_stress = 180;
train_stresses = setdiff(stress_all, test_stress);  % 除180外的所有应力
fprintf('训练应力: %s\n', mat2str(train_stresses'));
fprintf('测试应力: %d MPa\n', test_stress);

% 准备训练数据（按应力分组）
n_stress = length(train_stresses);
train_data = cell(n_stress, 1);
for i = 1:n_stress
    s = train_stresses(i);
    idx = raw(:,1) == s;
    time = raw(idx,2);
    strain = raw(idx,3);
    [time, sortIdx] = sort(time);
    strain = strain(sortIdx);
    stress_vec = s * ones(size(time));
    train_data{i} = struct('stress', stress_vec, 'time', time, 'strain', strain);
    fprintf('应力 %d MPa: %d 点, 时间 [%.1f, %.1f] h, 应变 [%.4f, %.4f]\n', ...
        s, length(time), min(time), max(time), min(strain), max(strain));
end

% 准备测试数据
idx_test = raw(:,1) == test_stress;
test_time = raw(idx_test,2);
test_strain = raw(idx_test,3);
[test_time, sortIdx] = sort(test_time);
test_strain = test_strain(sortIdx);
test_stress_vec = test_stress * ones(size(test_time));
fprintf('测试 %d MPa: %d 点, 时间 [%.1f, %.1f] h\n', ...
    test_stress, length(test_time), min(test_time), max(test_time));

%% 3. 指定用于架构选择的内部应力折
% 仅使用 170 和 190 作为留一折（内插），160 和 200 不用
internal_stresses = [170, 190];
% 检查这些应力是否在训练数据中
internal_stresses = internal_stresses(ismember(internal_stresses, train_stresses));
n_folds = length(internal_stresses);
fprintf('\n用于架构选择的内部应力折: %s\n', mat2str(internal_stresses));

%% 4. 候选架构列表
candidate_archs = {[2], [4], [8], [2 1], [2 4], [4 4], [2 2 4], [4 4 4]};
num_archs = length(candidate_archs);

%% 5. 留一内部应力交叉验证
cv_mse = zeros(num_archs, n_folds);

for a = 1:num_archs
    arch = candidate_archs{a};
    fprintf('\n评估架构: %s\n', mat2str(arch));
    
    for fold = 1:n_folds
        s_val = internal_stresses(fold);
        val_idx = find(train_stresses == s_val);
        train_idx = setdiff(1:n_stress, val_idx);
        
        % 构建训练集
        X_train = []; Y_train = [];
        for i = train_idx
            X_train = [X_train; train_data{i}.stress, train_data{i}.time];
            Y_train = [Y_train; train_data{i}.strain];
        end
        % 构建验证集
        X_val = [train_data{val_idx}.stress, train_data{val_idx}.time];
        Y_val = train_data{val_idx}.strain;
        
        % 归一化（仅用训练集参数）
        [X_train_norm, ps_X] = mapminmax(X_train', 0, 1);
        [Y_train_norm, ps_Y] = mapminmax(Y_train', 0, 1);
        X_val_norm = mapminmax('apply', X_val', ps_X);
        Y_val_norm = mapminmax('apply', Y_val', ps_Y);
        
        % 创建并训练网络
        net = feedforwardnet(arch, 'trainlm');
        net.trainParam.epochs = 1000;
        net.trainParam.goal = 1e-6;
        net.trainParam.showWindow = false;
        for j = 1:length(arch)
            net.layers{j}.transferFcn = 'tansig';
        end
        net.layers{end}.transferFcn = 'purelin';
        net.divideFcn = 'dividetrain';
        [net, tr] = train(net, X_train_norm, Y_train_norm);
        
        % 预测验证集
        Y_val_pred_norm = net(X_val_norm);
        Y_val_pred = mapminmax('reverse', Y_val_pred_norm, ps_Y);
        mse_val = mean((Y_val - Y_val_pred').^2);
        cv_mse(a, fold) = mse_val;
        
        fprintf('  留出 %d MPa: 验证 MSE = %.4e\n', s_val, mse_val);
    end
end

% 平均验证 MSE
mean_cv_mse = mean(cv_mse, 2);
[~, best_arch_idx] = min(mean_cv_mse);
best_arch = candidate_archs{best_arch_idx};

fprintf('\n===== 留一内部应力交叉验证结果（170 和 190 MPa）=====\n');
for a = 1:num_archs
    fprintf('架构 %-10s : 平均验证 MSE = %.4e\n', mat2str(candidate_archs{a}), mean_cv_mse(a));
end
fprintf('最佳架构: %s (平均验证 MSE = %.4e)\n', mat2str(best_arch), mean_cv_mse(best_arch_idx));

%% 6. 使用最佳架构在全部训练应力上训练最终模型
X_all_train = [];
Y_all_train = [];
for i = 1:n_stress
    X_all_train = [X_all_train; train_data{i}.stress, train_data{i}.time];
    Y_all_train = [Y_all_train; train_data{i}.strain];
end
[X_all_train_norm, ps_X_final] = mapminmax(X_all_train', 0, 1);
[Y_all_train_norm, ps_Y_final] = mapminmax(Y_all_train', 0, 1);

net_final = feedforwardnet(best_arch, 'trainlm');
net_final.trainParam.epochs = 1000;
net_final.trainParam.goal = 1e-6;
net_final.trainParam.showWindow = false;
for j = 1:length(best_arch)
    net_final.layers{j}.transferFcn = 'tansig';
end
net_final.layers{end}.transferFcn = 'purelin';
net_final.divideFcn = 'dividetrain';
[net_final, tr_final] = train(net_final, X_all_train_norm, Y_all_train_norm);

%% 7. 在 180 MPa 独立测试集上评估
X_test = [test_stress_vec, test_time]';
Y_test = test_strain;
X_test_norm = mapminmax('apply', X_test, ps_X_final);
Y_test_pred_norm = net_final(X_test_norm);
Y_test_pred = mapminmax('reverse', Y_test_pred_norm, ps_Y_final);
Y_test_pred = Y_test_pred';

R2 = 1 - sum((Y_test - Y_test_pred).^2) / sum((Y_test - mean(Y_test)).^2);
MAPE = mean(abs((Y_test - Y_test_pred) ./ Y_test)) * 100;
RMSE = sqrt(mean((Y_test - Y_test_pred).^2));

fprintf('\n===== 180 MPa 独立测试集评估 =====\n');
fprintf('R2   = %.4f\n', R2);
fprintf('MAPE = %.2f%%\n', MAPE);
fprintf('RMSE = %.4f\n', RMSE);

% 绘制对比图
figure;
plot(test_time, Y_test, 'b-', 'LineWidth', 1.5); hold on;
plot(test_time, Y_test_pred, 'r--', 'LineWidth', 1.5);
xlabel('Time (h)');
ylabel('Creep strain (%)');
legend('Experimental', 'BPNN prediction');
title('180 MPa creep curve prediction');
grid on;

%% 8. （可选）对最佳架构进行多种子统计（5个种子）
num_seeds = 5;
mse_per_seed = zeros(num_seeds, 1);
test_mse_per_seed = zeros(num_seeds, 1);

for seed = 1:num_seeds
    rng(seed);
    cv_mse_seed = zeros(n_folds, 1);
    for fold = 1:n_folds
        s_val = internal_stresses(fold);
        val_idx = find(train_stresses == s_val);
        train_idx = setdiff(1:n_stress, val_idx);
        
        X_train = []; Y_train = [];
        for i = train_idx
            X_train = [X_train; train_data{i}.stress, train_data{i}.time];
            Y_train = [Y_train; train_data{i}.strain];
        end
        X_val = [train_data{val_idx}.stress, train_data{val_idx}.time];
        Y_val = train_data{val_idx}.strain;
        
        [X_train_norm, ps_X] = mapminmax(X_train', 0, 1);
        [Y_train_norm, ps_Y] = mapminmax(Y_train', 0, 1);
        X_val_norm = mapminmax('apply', X_val', ps_X);
        Y_val_norm = mapminmax('apply', Y_val', ps_Y);
        
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
        
        Y_val_pred_norm = net(X_val_norm);
        Y_val_pred = mapminmax('reverse', Y_val_pred_norm, ps_Y);
        cv_mse_seed(fold) = mean((Y_val - Y_val_pred').^2);
    end
    mse_per_seed(seed) = mean(cv_mse_seed);
    
    % 同时记录 180 MPa 测试集 MSE
    X_all_train_norm = mapminmax('apply', X_all_train', ps_X);
    Y_all_train_norm = mapminmax('apply', Y_all_train', ps_Y);
    net_final_seed = feedforwardnet(best_arch, 'trainlm');
    net_final_seed.trainParam.epochs = 1000;
    net_final_seed.trainParam.goal = 1e-6;
    net_final_seed.trainParam.showWindow = false;
    for j = 1:length(best_arch)
        net_final_seed.layers{j}.transferFcn = 'tansig';
    end
    net_final_seed.layers{end}.transferFcn = 'purelin';
    net_final_seed.divideFcn = 'dividetrain';
    [net_final_seed, ~] = train(net_final_seed, X_all_train_norm, Y_all_train_norm);
    
    X_test_norm = mapminmax('apply', X_test, ps_X);
    Y_test_pred_norm = net_final_seed(X_test_norm);
    Y_test_pred_seed = mapminmax('reverse', Y_test_pred_norm, ps_Y)';
    test_mse_per_seed(seed) = mean((Y_test - Y_test_pred_seed).^2);
end

mean_cv_seed = mean(mse_per_seed);
std_cv_seed = std(mse_per_seed);
mean_test_seed = mean(test_mse_per_seed);
std_test_seed = std(test_mse_per_seed);

fprintf('\n===== 多种子统计（架构 %s, %d seeds）=====\n', mat2str(best_arch), num_seeds);
fprintf('验证集 MSE: %.4e ± %.4e (CV = %.2f%%)\n', ...
    mean_cv_seed, std_cv_seed, std_cv_seed/mean_cv_seed*100);
fprintf('180 MPa 测试集 MSE: %.4e ± %.4e\n', mean_test_seed, std_test_seed);