clc;
clear all;
close all;

%% Loading Experimental Data

dt = 0.001; 
stop_time = 25; 
DR = 0.99; % Fixed
f = 1;  %Fixed

%% System Identifications
%x = [m_p, c_p, k, m_s, c_s, mu*Nor, alpha_1, alpha_2]
x0 = [1.5, 53, 123.3, 3.2, 142.2, 0.08, 2.3, 0.3]; 
step_vector = [05, 0.5, 0.5, 0.1, 0.5, 1e-1, 0.1, 0.2];

options = optimoptions('fmincon',...
    'Display', 'iter-detailed',...
    'Algorithm', 'interior-point',...         
    'MaxIterations', 50,...                  
    'MaxFunctionEvaluations', 3000,...        
    'OptimalityTolerance', 1e-8,...           
    'StepTolerance', 1e-8,...                 
    'FiniteDifferenceStepSize', step_vector,... 
    'FiniteDifferenceType', 'central'); 

cost_function = @(x) calc_multi_mae(x, dt, f, DR, stop_time);

[x_opt, total_fval] = fmincon(cost_function, x0, [], [], [], [], [], [], [], options);

disp('Identified Parameters [m_p, c_p, k, m_s, c_s, mu*Nor, alpha_1, alpha_2]:');
disp(x_opt);

save('identified_params_multi_dataset.mat', 'x_opt', 'total_fval');


%% Plot
n_samples = round(stop_time/dt);
temp_data_100 = load('DR99f1V100.mat'); 
n_100 = min(n_samples, length(temp_data_100.t));
t_100 = temp_data_100.t(1:n_100);
x2_exp_100 = movmean(temp_data_100.outx2(1:n_100), 4); 

y_sim_100 = rk4_solver_inline(t_100, [0;0;0;0], dt, x_opt, 100, f, DR);
x2_sim_100 = y_sim_100(:,3) * 1e6;

figure('Name', 'Multi-Dataset ID Result: 100V');
plot(t_100, x2_exp_100, 'r', 'LineWidth', 1.5); hold on;
plot(t_100, x2_sim_100, 'b--', 'LineWidth', 1.2);
grid minor; grid on; 
title('Multi-Dataset Validation | V = 100V');
xlabel('Time [s]'); ylabel('x_2 [\mum]');
legend('Experimental', 'Model', 'Location', 'best');

% --- 2. Evaluate and Plot 140V ---
temp_data_140 = load('DR99f1V140.mat'); 
n_140 = min(n_samples, length(temp_data_140.t));
t_140 = temp_data_140.t(1:n_140);
x2_exp_140 = movmean(temp_data_140.outx2(1:n_140), 4); 

y_sim_140 = rk4_solver_inline(t_140, [0;0;0;0], dt, x_opt, 140, f, DR);
x2_sim_140 = y_sim_140(:,3) * 1e6;

figure('Name', 'Multi-Dataset ID Result: 140V');
plot(t_140, x2_exp_140, 'r', 'LineWidth', 1.5); hold on;
plot(t_140, x2_sim_140, 'b--', 'LineWidth', 1.2);
grid minor; grid on; 
title('Multi-Dataset Validation | V = 140V');
xlabel('Time [s]'); ylabel('x_2 [\mum]');
legend('Experimental', 'Model', 'Location', 'best');

%% Functions

function total_mae = calc_multi_mae(x, dt, f, DR, stop_time)
    total_mae = 0;
    train_voltages = [100, 140];
    train_files = {'DR99f1V100.mat', 'DR99f1V140.mat'};
    n_samples = round(stop_time/dt);
    
    for i = 1:length(train_voltages)
        current_amp = train_voltages(i);
        current_file = train_files{i};
        
        temp_data = load(current_file); 
        n = min(n_samples, length(temp_data.t));
        t_exp = temp_data.t(1:n);
        
        x2_exp = movmean(temp_data.outx2(1:n), 4); 
        
        y_sim = rk4_solver_inline(t_exp, [0;0;0;0], dt, x, current_amp, f, DR);
        x2_sim = y_sim(:,3) * 1e6; 
        
        T_total = t_exp(end) - t_exp(1);
        current_mae = (dt / T_total) * sum(abs(x2_sim - x2_exp));
        
        total_mae = total_mae + current_mae;
    end
end

% RK4 Solver
function y = rk4_solver_inline(t, y0, dt, x, amp, f, DR)
    N = length(t);
    y = zeros(N, 4);
    y(1,:) = y0';
    
    x1 = x(1); x2 = x(2); x3 = x(3);
    x4 = x(4); x5 = x(5); x6 = x(6);
    alpha1 = x(7); alpha2 = x(8);
    
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