clc; clear all; close all;

%% 1. Loading and Preprocessing Data
dt = 0.001; 
stop_time = 12; 
n_samples = round(stop_time/dt);
N_filter = 3;  
main_amp = 100; main_freq = 3; 

% Physical Constants (Converted to SI units)
global n_const kp_const;
n_const = 1;            % Constant n
kp_const = 1;      % 28 N/um to N/m

% Ordered file names (1-3 for Training, 4-6 for Validation)
file_names = { ...
    'DR975f3V100.mat', 'DR985f3V100.mat', 'DR99f3V100.mat', ... % Identification
    'DR97f3V100.mat', 'DR98f3V100.mat', 'DR995f3V100.mat' ...   % Validation
};

% Corresponding duty cycles for the new order
duty_values = [0.975, 0.985, 0.99, 0.97, 0.98, 0.995];

for i = 1:6
    test_configs(i).duty_saw = duty_values(i);
    test_configs(i).amp_saw = main_amp;
    test_configs(i).f = main_freq;
end

t_cell = cell(1,6); x2_exp_cell = cell(1,6);

for i = 1:6
    temp_data = load(file_names{i}); 
    n = min(n_samples, length(temp_data.t));
    t_cell{i} = temp_data.t(1:n);
    % Convert to meters
    x2_exp_cell{i} = movmean(temp_data.outx2(1:n), N_filter) * 1e-6; 
end

%% 2. System Identification (Training on 1:3, Validating on 4:6)
train_idx = [1, 2, 3]; % Training indices
val_idx = [4, 5, 6];   % Validation indices

% x0(7) is d33. Initial guess: 0.5 um/V -> 0.5e-6 m/V
x0 = [3617, 1646, 140482, 30, 82308, 6.32, 3.04]; 
step_vector = [0.05, 0.5, 0.5, 0.01, 0.5, 1e-4, 0.01];


options = optimoptions('fmincon', ...
    'Display', 'iter-detailed', ...
    'Algorithm', 'interior-point', ...
    'MaxIter', 150, ...
    'MaxFunctionEvaluations', 10000, ...
    'OptimalityTolerance', 1e-6);

% Cost function using your Integral MAE formula
global_cost_fun = @(x) calculate_total_mae(x, t_cell(train_idx), x2_exp_cell(train_idx), test_configs(train_idx), dt);

% Running fmincon with NO constraints (Unconstrained)
[x_opt, fval] = fmincon(global_cost_fun, x0, [], [], [], [], [], [], [], options);

fprintf('Optimization complete. Optimal parameters:\n');
disp(x_opt);

%% 3. Validation and Visualization
x2_sim_cell = cell(1,6);
for i = 1:6
    y_sim = rk4_solver(@(t, y) frictionODE_ident(t, y, x_opt, test_configs(i)), t_cell{i}, [0;0;0;0], dt);
    x2_sim_cell{i} = y_sim(:,3); 
end

figure('Name', 'Validation Dashboard (Integral MAE)', 'Position', [50, 50, 1000, 700]);
for i = 1:6
    % Calculate Integral MAE for plotting
    T_total = t_cell{i}(end) - t_cell{i}(1);
    current_mae = (dt / T_total) * sum(abs(x2_sim_cell{i} - x2_exp_cell{i}));
    
    subplot(3, 2, i);
    plot(t_cell{i}, x2_exp_cell{i}*1e6, 'r', 'LineWidth', 1.5); hold on;
    plot(t_cell{i}, x2_sim_cell{i}*1e6, 'b--', 'LineWidth', 1.2);
    grid on; 
    
    if ismember(i, train_idx)
        type_str = ' [Train]';
    else
        type_str = ' [Val]';
    end
    
    title(['Duty: ', num2str(test_configs(i).duty_saw), type_str, ' | MAE: ', num2str(current_mae*1e6, '%.2f'), ' \mum']);
    xlabel('Time [s]'); ylabel('Disp [\mum]');
    legend('Exp', 'Model', 'Location', 'best');
end

%% 4. Helper Functions
function total_mae = calculate_total_mae(x, t_train, x2_train, configs, dt)
    total_mae = 0;
    for k = 1:length(configs)
        y_sim = rk4_solver(@(t, y) frictionODE_ident(t, y, x, configs(k)), t_train{k}, [0;0;0;0], dt);
        x2_sim = y_sim(:, 3);
        if any(isnan(x2_sim)), total_mae = total_mae + 1e12; continue; end
        
        % Your exact Integral MAE formulation
        T_total = t_train{k}(end) - t_train{k}(1);
        mae_k = (dt / T_total) * sum(abs(x2_sim - x2_train{k}));
        total_mae = total_mae + mae_k; 
    end
end

function y = rk4_solver(ode_fun, t, y0, dt)
    N = length(t); y = zeros(N, 4); y(1,:) = y0';
    for i = 1:(N-1)
        k1 = ode_fun(t(i), y(i,:)');
        k2 = ode_fun(t(i)+dt/2, y(i,:)'+(dt/2)*k1);
        k3 = ode_fun(t(i)+dt/2, y(i,:)'+(dt/2)*k2);
        k4 = ode_fun(t(i)+dt, y(i,:)'+dt*k3);
        y(i+1,:) = (y(i,:)' + (dt/6)*(k1 + 2*k2 + 2*k3 + k4))';
    end
end

function dydt = frictionODE_ident(t, y, x, cfg)
    global n_const kp_const;
    V_t = (cfg.amp_saw / 2) * (sawtooth(2 * pi * cfg.f * t, cfg.duty_saw) + 1);
    
    % x(7) is d33
    d33 = x(7);
    alpha = n_const * kp_const * d33;
    
    ff = x(6) * sign(y(2) - y(4));
    dydt = [y(2); (alpha*V_t - x(2)*y(2) - x(3)*y(1) - ff)/x(1); y(4); (ff - x(5)*y(4))/x(4)];
end