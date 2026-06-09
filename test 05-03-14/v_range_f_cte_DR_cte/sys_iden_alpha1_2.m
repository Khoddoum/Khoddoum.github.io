clc;
clear all;
close all;

%% Loading Experimental Data

dt = 0.001; 
stop_time = 25; 
n_samples = round(stop_time/dt);  

DR = 0.99;
f = 1; % Frequency [Hz]
amp = 100;

temp_data = load('DR99f1V100.mat'); 
n = min(n_samples, length(temp_data.t));
t_st = temp_data.t(1:n);

% Applying Filter
x2_exp = movmean(temp_data.outx2(1:n), 4); 

%% System Identifications
%x = [m_p, c_p, k, m_s, c_s, mu*N, alpha_1, alpha_2]
x0 = [2.5, 75, 125, 3.5, 144.5, 0.008, 2, 0.5]; 
step_vector = [0.05, 0.5, 0.5, 0.01, 0.5, 1e-5, 0.01, 0.02];
 
options = optimoptions('fmincon',...
    'Display', 'iter-detailed',...
    'Algorithm', 'interior-point',...         
    'MaxIterations', 120,...                  
    'MaxFunctionEvaluations', 3000,...        
    'OptimalityTolerance', 1e-6,...           
    'StepTolerance', 1e-12,...                 
    'FiniteDifferenceStepSize', step_vector,... 
    'FiniteDifferenceType', 'forward'); 

cost_fun = @(x) calc_single_mae(x, t_st, x2_exp, dt, amp, f, DR);

[x_opt, fval] = fmincon(cost_fun, x0, [], [], [], [], [], [], [], options);

disp('m_p, c_p, k, m_s, c_s, mu*Nor, alpha_1, alpha_2');
disp(x_opt);

%% Plot

y_sim = rk4_solver_inline(t_st, [0;0;0;0], dt, x_opt, amp, f, DR);
x2_sim = y_sim(:,3) * 1e6;

figure('Name', 'System Identification Result');
plot(t_st, x2_exp, 'r', 'LineWidth', 1.5);
hold on;
plot(t_st, x2_sim, 'b--', 'LineWidth', 1.2);
grid minor;
grid on; 
% title(['F_p = \alpha_1 V + \alpha_2 V^2 | Final MAE: ', num2str(fval, '%.2f'), ' \mum']);
xlabel('Time [s]'); ylabel('x_2 [\mum]');
legend('Experimental', 'Model', 'Location', 'best');

%% Validation on Other Datasets

% Define the validation voltages and corresponding filenames
val_voltages = [90, 110, 120, 130, 140];
val_files = {'DR99f1V90.mat', 'DR99f1V110.mat', 'DR99f1V120.mat', 'DR99f1V130.mat', 'DR99f1V140.mat'};

for i = 1:length(val_voltages)
    current_amp = val_voltages(i);
    current_file = val_files{i};
    
    % Check if file exists to prevent errors
    if isfile(current_file)
        % Load Validation Data
        val_data = load(current_file); 
        n_v = min(n_samples, length(val_data.t));
        t_val = val_data.t(1:n_v);
        
        % Applying same filter to experimental validation data
        x2_exp_val = movmean(val_data.outx2(1:n_v), 4); 
        
        % Simulate Model using Identified Parameters (x_opt)
        y_sim_val = rk4_solver_inline(t_val, [0;0;0;0], dt, x_opt, current_amp, f, DR);
        x2_sim_val = y_sim_val(:,3) * 1e6;
        
        % Calculate MAE for validation dataset
        T_total_val = t_val(end) - t_val(1);
        val_mae = (dt / T_total_val) * sum(abs(x2_sim_val - x2_exp_val));
        
        % Plotting in Separate Figures
        figure('Name', ['Validation: V = ', num2str(current_amp), 'V']);
        plot(t_val, x2_exp_val, 'r', 'LineWidth', 1.5); hold on;
        plot(t_val, x2_sim_val, 'b--', 'LineWidth', 1.2);
        grid minor; grid on; 
        
        title(['Validation | DR=0.99, f=1Hz, V=', num2str(current_amp), 'V | MAE: ', num2str(val_mae, '%.2f'), ' \mum']);
        xlabel('Time [s]'); ylabel('x_2 [\mum]');
        legend('Experimental', 'Model (Identified)', 'Location', 'best');
    else
        fprintf('Warning: File %s not found in the current directory. Skipping...\n', current_file);
    end
end

%%  Functions
% Error Function
function mae = calc_single_mae(x, t, x2_data, dt, amp, f, DR)
    y_sim = rk4_solver_inline(t, [0;0;0;0], dt, x, amp, f, DR);
    x2_sim = y_sim(:,3) * 1e6; 
    T_total = t(end) - t(1);
    mae = (dt / T_total) * sum(abs(x2_sim - x2_data)); % Error Function
end

% RK4

function y = rk4_solver_inline(t, y0, dt, x, amp, f, DR)
    N = length(t);
    y = zeros(N, 4);
    y(1,:) = y0';
    
    x1 = x(1);
    x2 = x(2);
    x3 = x(3);
    x4 = x(4);
    x5 = x(5);
    x6 = x(6);
    alpha1 = x(7);
    alpha2 = x(8);
    
    amp_half = amp / 2;
    two_pi_f = 2 * pi * f; 
    dt_half = dt / 2;
    dt_sixth = dt / 6;
    
    for i = 1:(N-1)
        ti = t(i); yi = y(i,:)';
        
        V1 = amp_half * (sawtooth(two_pi_f * ti, DR) + 1);
        Fp1 = alpha1 * V1 + alpha2 * (V1^2);
        ff1 = x6 * sign(yi(2) - yi(4));
        k1 = [yi(2); (Fp1 - x2*yi(2) - x3*yi(1) - ff1)/x1; yi(4); (ff1 - x5*yi(4))/x4];
        
        yi_k2 = yi + dt_half * k1;
        V2 = amp_half * (sawtooth(two_pi_f * (ti + dt_half), DR) + 1);
        Fp2 = alpha1 * V2 + alpha2 * (V2^2);
        ff2 = x6 * sign(yi_k2(2) - yi_k2(4));
        k2 = [yi_k2(2); (Fp2 - x2*yi_k2(2) - x3*yi_k2(1) - ff2)/x1; yi_k2(4); (ff2 - x5*yi_k2(4))/x4];
        
        yi_k3 = yi + dt_half * k2;
        ff3 = x6 * sign(yi_k3(2) - yi_k3(4));
        k3 = [yi_k3(2); (Fp2 - x2*yi_k3(2) - x3*yi_k3(1) - ff3)/x1; yi_k3(4); (ff3 - x5*yi_k3(4))/x4];
        
        yi_k4 = yi + dt * k3;
        V4 = amp_half * (sawtooth(two_pi_f * (ti + dt), DR) + 1);
        Fp4 = alpha1 * V4 + alpha2 * (V4^2);
        ff4 = x6 * sign(yi_k4(2) - yi_k4(4));
        k4 = [yi_k4(2); (Fp4 - x2*yi_k4(2) - x3*yi_k4(1) - ff4)/x1; yi_k4(4); (ff4 - x5*yi_k4(4))/x4];
        
        y(i+1,:) = (yi + dt_sixth * (k1 + 2*k2 + 2*k3 + k4))';
    end
end