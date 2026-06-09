clc; clear all; close all;

%% 1. Loading and Preprocessing Data
dt = 0.001; 
stop_time = 10; 
n_samples = round(stop_time/dt);
N_filter = 5;  
main_amp = 100; main_freq = 3; 

% Configure duty cycles for 6 datasets
duty_values = [0.985, 0.995, 0.97, 0.975, 0.98, 0.99];
for i = 1:6
    test_configs(i).duty_saw = duty_values(i);
    test_configs(i).amp_saw = main_amp;
    test_configs(i).f = main_freq;
end

file_names = {'DR985f3V100.mat', 'DR995f3V100.mat', 'DR97f3V100.mat', 'DR975f3V100.mat', 'DR98f3V100.mat', 'DR99f3V100.mat'};
t_cell = cell(1,6); x2_exp_cell = cell(1,6);

for i = 1:6
    temp_data = load(file_names{i}); 
    n = min(n_samples, length(temp_data.t));
    t_cell{i} = temp_data.t(1:n);
    % Experimental data kept in microns for direct comparison
    x2_exp_cell{i} = movmean(temp_data.outx2(1:n), N_filter); 
end

%% 2. System Identification (Training on DR985f3V100 only)
train_idx = 1;         % Only the first dataset (DR985) is used for training
val_idx = [2, 3, 4, 5, 6]; % Rest are used for validation

% Initial guess
x0 = [1.2, 89, 103000, 0.9, 80, 0.8, 6]; 
step_vector = [0.05, 0.5, 0.5, 0.01, 0.5, 1e-4, 0.01];

% Optimization settings for fmincon (Unconstrained)
options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'interior-point', ...
    'MaxIterations', 100, ...
    'MaxFunctionEvaluations', 10000, ...
    'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-12, 'FiniteDifferenceStepSize', step_vector);

% Cost function definition
global_cost_fun = @(x) calculate_total_mae(x, t_cell(train_idx), x2_exp_cell(train_idx), test_configs(train_idx), dt);

% Execution using fmincon without linear or non-linear constraints
fprintf('Running unconstrained fmincon optimization...\n');
[x_opt, fval] = fmincon(global_cost_fun, x0, [], [], [], [], [], [], [], options);

fprintf('Optimization complete. Optimal parameters:\n');
disp(x_opt);

%% 3. Validation and Visualization (Dashboard)
x2_sim_cell = cell(1,6);
for i = 1:6
    % High-performance simulation
    y_sim = rk4_solver_inline(t_cell{i}, [0;0;0;0], dt, x_opt, test_configs(i));
    x2_sim_cell{i} = y_sim(:,3) * 1e6; % Convert solver output (meters) to microns
end

figure('Name', 'System Identification & Validation Dashboard', 'Position', [50, 50, 1100, 750]);
for i = 1:6
    T_total = t_cell{i}(end) - t_cell{i}(1);
    % Corrected integral MAE calculation in microns
    current_mae = (dt / T_total) * sum(abs(x2_sim_cell{i} - x2_exp_cell{i}));
    
    subplot(3, 2, i);
    plot(t_cell{i}, x2_exp_cell{i}, 'r', 'LineWidth', 1.5); hold on;
    plot(t_cell{i}, x2_sim_cell{i}, 'b--', 'LineWidth', 1.2);
    grid on; 
    
    if i == train_idx
        title_suffix = ' [Train]';
    else
        title_suffix = ' [Val]';
    end
    
    title(['Duty: ', num2str(test_configs(i).duty_saw), title_suffix, ' | MAE: ', num2str(current_mae, '%.2f'), ' \mum']);
    xlabel('Time [s]'); ylabel('Disp [\mum]');
    legend('Exp', 'Model', 'Location', 'best');
end

%% 4. Helper Functions
function total_mae = calculate_total_mae(x, t_train, x2_train, configs, dt)
    total_mae = 0;
    for k = 1:length(configs)
        % Fast inline RK4 solver execution
        y_sim = rk4_solver_inline(t_train{k}, [0;0;0;0], dt, x, configs(k));
        x2_sim = y_sim(:, 3) * 1e6; % Convert to microns to match x2_train
              
        T_total = t_train{k}(end) - t_train{k}(1);
        mae_k = (dt / T_total) * sum(abs(x2_sim - x2_train{k}));
        
        % Structural Fix: Accumulate the errors across datasets
        total_mae = total_mae + mae_k; 
    end
end

% Optimized inline RK4 solver removing functional overheads for fmincon speed
function y = rk4_solver_inline(t, y0, dt, x, cfg)
    N = length(t); y = zeros(N, 4); y(1,:) = y0';
    x1 = x(1); x2 = x(2); x3 = x(3); x4 = x(4); x5 = x(5); x6 = x(6); alpha = x(7);
    amp_half = cfg.amp_saw / 2; two_pi_f = 2 * pi * cfg.f;
    duty = cfg.duty_saw; dt_half = dt / 2; dt_sixth = dt / 6;
    
    for i = 1:(N-1)
        ti = t(i); yi = y(i,:)';
        
        V1 = amp_half * (sawtooth(two_pi_f * ti, duty) + 1);
        ff1 = x6 * sign(yi(2) - yi(4));
        k1 = [yi(2); (alpha*V1 - x2*yi(2) - x3*yi(1) - ff1)/x1; yi(4); (ff1 - x5*yi(4))/x4];
        
        yi_k2 = yi + dt_half * k1;
        V2 = amp_half * (sawtooth(two_pi_f * (ti + dt_half), duty) + 1);
        ff2 = x6 * sign(yi_k2(2) - yi_k2(4));
        k2 = [yi_k2(2); (alpha*V2 - x2*yi_k2(2) - x3*yi_k2(1) - ff2)/x1; yi_k2(4); (ff2 - x5*yi_k2(4))/x4];
        
        yi_k3 = yi + dt_half * k2;
        ff3 = x6 * sign(yi_k3(2) - yi_k3(4));
        k3 = [yi_k3(2); (alpha*V2 - x2*yi_k3(2) - x3*yi_k3(1) - ff3)/x1; yi_k3(4); (ff3 - x5*yi_k3(4))/x4];
        
        yi_k4 = yi + dt * k3;
        V4 = amp_half * (sawtooth(two_pi_f * (ti + dt), duty) + 1);
        ff4 = x6 * sign(yi_k4(2) - yi_k4(4));
        k4 = [yi_k4(2); (alpha*V4 - x2*yi_k4(2) - x3*yi_k4(1) - ff4)/x1; yi_k4(4); (ff4 - x5*yi_k4(4))/x4];
        
        y(i+1,:) = (yi + dt_sixth * (k1 + 2*k2 + 2*k3 + k4))';
    end
end