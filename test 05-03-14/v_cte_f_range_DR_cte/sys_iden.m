clc;
clear all;
close all;
%% 1. Configuration & Data Setup
files = {'DR99f1V100.mat', 'DR99f3V100.mat', 'DR99f5V100.mat', 'DR99f7V100.mat', 'DR99f9V100.mat'};
t_data = cell(1, 5); x2_data = cell(1, 5);
dt = 0.001; stop_time = 8; n_samples = round(stop_time/dt);

for i = 1:5
    d = load(files{i});
    t_data{i} = d.t(1:n_samples);
    x2_data{i} = movmean(d.outx2(1:n_samples), 2);
end

%% 2. Optimization Setup
LB = [3.5, 50, 40, 0.4, 40, 0.0001, 1,   0.8, 2.5, 4.5, 6.5, 8.5, 0.90];
UB = [12, 250, 250, 3.5, 250, 0.1, 50,   1.2, 3.5, 5.5, 7.5, 9.5, 0.99];
options = optimoptions('gamultiobj', 'Display', 'iter', 'PopulationSize', 40, 'MaxGenerations', 20);

% Multi-objective optimization
[x_opt_set, fval] = gamultiobj(@(x) my_multiobjective_function(x, t_data, x2_data, dt), 13, [], [], [], [], LB, UB, options);

%% 3. Post-Processing & Validation
% Extract the best solution
x_final = x_opt_set(1, :); 

% Display identified physical parameters
fprintf('--- Identified Physical Parameters ---\n');
fprintf('Mass (mp): %.4f\nStiffness (k): %.4f\nFriction Coeff (alpha): %.4f\n', x_final(1), x_final(3), x_final(7));
disp('Identified Frequencies:');
disp(x_final(8:12));

% Visualization of results
figure('Name', 'Final System Identification Results');
for i = 1:5
    p.f = x_final(7+i); p.duty_saw = x_final(13); p.amp_saw = 100;
    y_sim = rk4_solver(@(t, y) frictionODE(t, y, x_final(1:7), p), t_data{i}, [0;0;0;0], dt);
    
    subplot(5, 1, i);
    plot(t_data{i}, x2_data{i}, 'b', t_data{i}, y_sim(:,3), 'r--');
    title(['Frequency Fit: ', num2str(x_final(7+i), '%.2f'), ' Hz']);
    legend('Experimental', 'Simulated'); grid on;
end

%% 4. Helper Functions
function F = my_multiobjective_function(x, t_data, x2_data, dt)
    F = zeros(1, 5);
    for i = 1:5
        p.f = x(7+i); p.duty_saw = x(13); p.amp_saw = 100;
        y_s = rk4_solver(@(t, y) frictionODE(t, y, x(1:7), p), t_data{i}, [0;0;0;0], dt);
        F(i) = sum((y_s(:,3) - x2_data{i}).^2) / sum(x2_data{i}.^2);
    end
end

function dydt = frictionODE(t, y, x_phys, p)
    ff = x_phys(6) * tanh(200 * (y(2) - y(4))); 
    Vt = (p.amp_saw / 2) * (sawtooth(2 * pi * p.f * t, p.duty_saw) + 1);
    dydt = [y(2); (x_phys(7)*Vt - x_phys(2)*y(2) - x_phys(3)*y(1) - ff)/x_phys(1); y(4); (ff - x_phys(5)*y(4))/x_phys(4)];
end

function y = rk4_solver(ode, t, y0, dt)
    N = length(t); y = zeros(N, 4); y(1, :) = y0';
    for i = 1:(N-1)
        k1 = ode(t(i), y(i,:)');
        k2 = ode(t(i)+dt/2, y(i,:)'+(dt/2)*k1);
        k3 = ode(t(i)+dt/2, y(i,:)'+(dt/2)*k2);
        k4 = ode(t(i)+dt, y(i,:)'+dt*k3);
        y(i+1,:) = (y(i,:)' + (dt/6)*(k1 + 2*k2 + 2*k3 + k4))';
    end
end