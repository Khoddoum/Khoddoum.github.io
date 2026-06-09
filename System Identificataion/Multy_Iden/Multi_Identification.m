clc;
clear all;
close all;

%% Loading Experimental Data
dt = 0.001; 
stop_time = 5; 
n_samples = round(stop_time/dt);
N_filter = 2;  % Applying Movemean Filter

test_configs(1).amp_saw = 80;
test_configs(1).f = 1;          
test_configs(1).duty_saw = 0.99;

test_configs(2).amp_saw = 80;
test_configs(2).f = 3;          
test_configs(2).duty_saw = 0.99;

file_names = {'DR99F1V80.mat', 'DR99F3V80.mat'};

t_cell = cell(1, 2);
x2_exp_data_cell = cell(1, 2);

for i = 1:length(file_names)
    temp_data = load(file_names{i}); 
    t_cell{i} = temp_data.t(1:n_samples);
    
    outx2_st = temp_data.x2(1:n_samples);
    outx2_st_fil = movmean(outx2_st, N_filter); %Movemean
    x2_exp_data_cell{i} = outx2_st_fil; % unit: micron
    assignin('base', sprintf('t_dataset%d', i), t_cell{i});
    assignin('base', sprintf('x2_dataset%d', i), x2_exp_data_cell{i});
end

%% System Identification
%      [   mp,    cp,         k,        ms,            cs,   miu_Nor,     alpha]
x0  = [2.04,   186,       480,      1.33,          147,    0.0098,       7.8]; 
LB  = [ 1.5,   150,       400,       0.5,          100,    0.0001,       7.0]; % Narrowed around 7.8
UB  = [  10,   320,       650,       4.5,          300,      0.02,       8.5]; % Narrowed around 7.8

step_vector = [0.005, 0.05,  0.05, 0.001,  0.05,  1e-5,  0.001];


options = optimoptions('fmincon',...
    'Display', 'iter-detailed',...
    'Algorithm', 'interior-point',...         
    'MaxIterations', 120,...                  
    'MaxFunctionEvaluations', 3000,...        
    'OptimalityTolerance', 1e-6,...           
    'StepTolerance', 1e-12,...                 
    'FiniteDifferenceStepSize', step_vector,... 
    'FiniteDifferenceType', 'central');       

global_cost_fun = @(x) my_global_cost_function(x, t_cell, x2_exp_data_cell, test_configs, dt);

[x_opt, fval] = fmincon(global_cost_fun, x0, [], [], [], [], LB, UB, [], options);

fprintf('mp = %f\ncp = %f\nk  = %f\nms = %f\ncs = %f\nmiu*Nor = %f\nalpha = %f\n', ...
x_opt(1),x_opt(2),x_opt(3),x_opt(4),x_opt(5),x_opt(6),x_opt(7));

%% Processing
[~, idx_st_2] = min(abs(t_cell{1} - 2)); 

x2_sim_micron_cell = cell(1,2);
fit_exp_line_cell = cell(1,2);
fit_num_line_cell = cell(1,2);

for i = 1:2
    y_sim = rk4_solver(@(t, y) frictionODE_ident(t, y, x_opt, test_configs(i)), t_cell{i}, [0;0;0;0], dt);
    x2_sim_micron_cell{i} = y_sim(:,3).*1e6; % convert to micron
    
    t_pfit = t_cell{i}(idx_st_2:end);
    x2_exp_pfit = x2_exp_data_cell{i}(idx_st_2:end);
    x2_sim_pfit = x2_sim_micron_cell{i}(idx_st_2:end);
    
    b_exp = polyfit(t_pfit, x2_exp_pfit, 1);
    v_avg_exp_vec(i) = b_exp(1);
    fit_exp_line_cell{i} = polyval(b_exp, t_cell{i});
    
    b_num = polyfit(t_pfit, x2_sim_pfit, 1);
    v_avg_num_vec(i) = b_num(1);
    fit_num_line_cell{i} = polyval(b_num, t_cell{i});
    
    assignin('base', sprintf('x2_sim_dataset%d', i), x2_sim_micron_cell{i});
    assignin('base', sprintf('fit_exp_line%d', i), fit_exp_line_cell{i});
    assignin('base', sprintf('fit_num_line%d', i), fit_num_line_cell{i});
    
    fprintf('Dataset %d (%d Hz) -> v_avg_exp = %f, v_avg_num = %f [micron/s]\n', ...
    i, test_configs(i).f, v_avg_exp_vec(i), v_avg_num_vec(i));
end

%% Plot
figure();
plot(t_cell{1}, x2_exp_data_cell{1}, 'b', 'LineWidth', 1.5);
hold on;
plot(t_cell{1}, x2_sim_micron_cell{1}, 'k--', 'LineWidth', 1.5);
plot(t_cell{2}, x2_exp_data_cell{2}, 'r', 'LineWidth', 1.5);
plot(t_cell{2}, x2_sim_micron_cell{2}, 'm--', 'LineWidth', 1.5);
grid minor;
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('Displacement [$\mu$m]', 'Interpreter', 'latex');
title('Experimental & Identified System [RK4]', 'Interpreter', 'latex');
legend('exp 1Hz', 'sim 1Hz', 'exp 3Hz', 'sim 3Hz', 'Location', 'best');

figure();
plot(t_cell{1}, fit_exp_line_cell{1}, 'b:', 'LineWidth', 1.2);
hold on;
plot(t_cell{1}, fit_num_line_cell{1}, 'c-.', 'LineWidth', 1.2);
plot(t_cell{2}, fit_exp_line_cell{2}, 'r:',  'LineWidth', 1.2);
plot(t_cell{2}, fit_num_line_cell{2}, 'm-.', 'LineWidth', 1.2);

grid minor;
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('Average Trend [$\mu$m]', 'Interpreter', 'latex');
title('Polyfit Average Velocity Slopes Comparison', 'Interpreter', 'latex');
legend('Exp Slope 1Hz', 'Sim Slope 1Hz', 'Exp Slope 3Hz', 'Sim Slope 3Hz', 'Location', 'best');


% figure();
% [~, idx_last2_1] = min(abs(t_cell{1} - (stop_time - 2)));
% [~, idx_last2_2] = min(abs(t_cell{2} - (stop_time - 2)));
% 
% plot(t_cell{1}(idx_last2_1:end), x2_exp_data_cell{1}(idx_last2_1:end), 'b', 'LineWidth', 1.5);
% hold on;
% plot(t_cell{1}(idx_last2_1:end), x2_sim_micron_cell{1}(idx_last2_1:end), 'c--', 'LineWidth', 1.5);
% plot(t_cell{2}(idx_last2_2:end), x2_exp_data_cell{2}(idx_last2_2:end), 'r', 'LineWidth', 1.5);
% plot(t_cell{2}(idx_last2_2:end), x2_sim_micron_cell{2}(idx_last2_2:end), 'm--', 'LineWidth', 1.5);
% 
% grid minor;
% xlabel('Time [s]', 'Interpreter', 'latex');
% ylabel('Displacement [$\mu$m]', 'Interpreter', 'latex');
% title('Zoomed View of Steady-State Ripples (Last 2 Seconds)', 'Interpreter', 'latex');
% legend('Exp 1Hz', 'Sim 1Hz', 'Exp 3Hz', 'Sim 3Hz', 'Location', 'best');

%% Functions

function total_score = my_global_cost_function(x, t_cell, x2_exp_data_cell, test_configs, dt)
    total_score = 0;
    y0 = [0; 0; 0; 0];
    w = [1,5];
    for i = 1:2
        y_sim = rk4_solver(@(t, y) frictionODE_ident(t, y, x, test_configs(i)), t_cell{i}, y0, dt);
        x2_sim_micron = y_sim(:, 3).*1e6; 
        
        T_total = t_cell{i}(end) - t_cell{i}(1);
        max_val = max(abs(x2_exp_data_cell{i}));
        if max_val == 0, max_val = 1; end
        
        mae = (dt / T_total) * sum(abs(x2_sim_micron - x2_exp_data_cell{i}));
        total_score = total_score + w(i)*(mae/max_val);
    end
end
% RK4 Solver 
function y = rk4_solver(ode_fun, t, y0, dt)
    N = length(t);
    y = zeros(N, length(y0));
    y(1,:) = y0';
    for i = 1:(N-1)
        ti = t(i); yi = y(i,:)';
        k1 = ode_fun(ti, yi);
        k2 = ode_fun(ti + dt/2, yi + (dt/2)*k1);
        k3 = ode_fun(ti + dt/2, yi + (dt/2)*k2);
        k4 = ode_fun(ti + dt, yi + dt*k3);
        y(i+1,:) = (yi + (dt/6)*(k1 + 2*k2 + 2*k3 + k4))';
    end
end

% Equation of Motion
function dydt = frictionODE_ident(t,y,x,input_params)
    mp = x(1);
    cp = x(2);
    k  = x(3);
    ms = x(4);
    cs = x(5);
    miu_Nor = x(6);
    alpha = x(7); 
    y2 = y(2);
    y4 = y(4);
    
    V_t = (input_params.amp_saw / 2) * (sawtooth(2 * pi * input_params.f * t, input_params.duty_saw) + 1);
    Fp = alpha * V_t;
    ff = miu_Nor * sign(y2 - y4);
    
    dydt = zeros(4,1);
    dydt(1) = y2;                               
    dydt(2) = (Fp - cp*y2 - k*y(1) - ff) / mp;  
    dydt(3) = y4;                               
    dydt(4) = (ff - cs*y4) / ms;                
end