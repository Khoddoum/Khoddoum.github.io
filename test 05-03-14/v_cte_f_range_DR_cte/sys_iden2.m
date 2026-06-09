clc; clear all; close all;

%% 1. Loading and Preprocessing Data (Target: DR99f1V100)
dt = 0.001; 
stop_time = 12; 
n_samples = round(stop_time/dt);
N_filter = 3;  

% Fixed operating parameters for the specific file
fixed_duty = 0.99;
fixed_freq = 1;
fixed_voltage = 100; 

file_names = {'DR99f1V100.mat'};
voltages = [fixed_voltage];

test_configs(1).duty_saw = fixed_duty;
test_configs(1).f = fixed_freq;
test_configs(1).amp_saw = fixed_voltage; 

t_cell = cell(1,1); x2_exp_cell = cell(1,1);

% Load the single targeted data file
temp_data = load(file_names{1}); 
n = min(n_samples, length(temp_data.t));
t_cell{1} = temp_data.t(1:n);
% Experimental data remains in microns
x2_exp_cell{1} = movmean(temp_data.outx2(1:n), N_filter); 

train_idx = 1; 
global_cost_fun = @(x) calculate_total_mae(x, t_cell(train_idx), x2_exp_cell(train_idx), test_configs(train_idx), dt);

%% 2. Particle Swarm Optimization (Unconstrained, MaxIterations = 50)
options_pso = optimoptions('particleswarm', ...
    'SwarmSize', 500, ...
    'MaxIterations', 50, ...
    'MaxStallIterations', 10, ...
    'FunctionTolerance', 1e-12, ...
    'UseParallel', true, ...
    'Display', 'iter', ...
    'InertiaRange', [0.4 0.9], ...
    'SelfAdjustmentWeight', 1.6, ...
    'SocialAdjustmentWeight', 1.4, ...
    'MinNeighborsFraction', 0.1);

fprintf('Running PSO for identification on DR99f1V100...\n');
[x_opt_pso, fval_pso] = particleswarm(global_cost_fun, 7, [], [], options_pso);

fprintf('\nOptimization complete. Identified parameters:\n');
disp(x_opt_pso);

%% 3. Validation and Visualization
x2_sim_cell = cell(1,1);
% Simulate and convert meter (output of solver) to microns
y_sim = rk4_solver_inline(t_cell{1}, [0;0;0;0], dt, x_opt_pso, test_configs(1));
x2_sim_cell{1} = y_sim(:,3) * 1e6; 

T_total = t_cell{1}(end) - t_cell{1}(1);
current_mae = (dt / T_total) * sum(abs(x2_sim_cell{1} - x2_exp_cell{1}));

fig_title = ['Voltage: ', num2str(test_configs(1).amp_saw), 'V [Identification]'];
figure('Name', fig_title);
plot(t_cell{1}, x2_exp_cell{1}, 'r', 'LineWidth', 1.5); hold on;
plot(t_cell{1}, x2_sim_cell{1}, 'b--', 'LineWidth', 1.2);
grid on; 
title([fig_title, ' | MAE: ', num2str(current_mae, '%.2f'), ' \mum']);
xlabel('Time [s]'); ylabel('Displacement [\mum]');
legend('Experimental', 'Model', 'Location', 'best');

%% 4. Helper Functions
function total_mae = calculate_total_mae(x, t_train, x2_train, configs, dt)
    total_mae = 0;
    for k = 1:length(configs)
        y_sim = rk4_solver_inline(t_train{k}, [0;0;0;0], dt, x, configs(k));
        x2_sim = y_sim(:, 3) * 1e6; % Convert simulation (m) to microns to match experimental data
        
        if any(isnan(x2_sim)), total_mae = total_mae + 1e12; continue; end
        
        T_total = t_train{k}(end) - t_train{k}(1);
        total_mae = total_mae + (dt / T_total) * sum(abs(x2_sim - x2_train{k}));
    end
end

function y = rk4_solver_inline(t, y0, dt, x, cfg)
    N = length(t); y = zeros(N, 4); y(1,:) = y0';
    x1 = x(1); x2 = x(2); x3 = x(3); x4 = x(4);
    x5 = x(5); x6 = x(6); alpha = x(7);
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