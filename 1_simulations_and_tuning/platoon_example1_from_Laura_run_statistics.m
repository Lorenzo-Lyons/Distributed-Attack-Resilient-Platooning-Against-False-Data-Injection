clc
close all
clear all
% Platooning example
Dt = 0.15;    % sampling period
beta = -0.1; % velocity loss caused by friction
d_opt = 1;   % desired distance between vehicles
kp = 0.2;    % proportional gain of the forward-and-reverse-looking PD control 
kd = 0.3;    % derivative gain of the forward-and-reverse-looking PD control 
% Control system with already implemented the primary control action:
% x+ = Fx+Gw, where x is the new state in the error dynamics and w is the
% secondary control action 

n_follower_vehicles = 2;


% F = [1  0 -Dt           Dt           0;
%      0  1  0           -Dt           Dt;
%      kp 0  (1+beta)-kd  kd           0;
%     -kp kp kd          (1+beta)-2*kd kd;
%      0 -kp 0            kd           (1+beta)-kd];
% G = [zeros(2,3);
%      Dt*eye(3)];
% n = 5;
% m = 3;


% 
[F,G,n,m] = build_FG(Dt,n_follower_vehicles,beta,kp,kd);







% Check the eigenvalues of F
eigenvalues = eig(F);
if all(abs(eigenvalues) < 1)
    disp('Eigenvalues are inside the unit circle');
end
disp(['Eigenvalues of F: ', mat2str(round(eigenvalues,2))]);

% Neatly print F
fprintf('\nMatrix F:\n');
disp(F);

% Neatly print G
fprintf('\nMatrix G:\n');
disp(G);



% Upper and lower bounds on the secondary control action wi

u_max_actuators = 4.905; % this is the real max acc capacity
v_d = 25; % desired platooning velocity

u_max = 7.848; 
u_min = -u_max;

Ub = ones(1,m) * u_max;
Lb = ones(1,m) *-u_max;

gamma = u_max^2; 

R = eye (m,m) * 1/gamma;



b = -6; % all the same so ok

dangerous_indices = [1, 2];  % Example: constraints on x1 and x2
num_constraints = length(dangerous_indices);

C = zeros(n, num_constraints);  % Each column will be a c_i vector

for i = 1:num_constraints
    C(dangerous_indices(i), i) = 1;  % c_i = unit vector in i-th dangerous direction
end











%a = 0.86;
% Initialize best objective value and solution
best_obj = Inf;
best_P = [];
best_a = NaN;

% Initialize arrays to store results
a_vals = [];
obj_vals = [];

% optimal a values
% n_follower_vehicles = 2  --> a = 0.888
% n_follower_vehicles = 10 --> a = 0.955

for a = [0.888] %linspace(0.95,0.99,20) % [0.955] %[0.888] % [0.955] %linspace(0.93,0.99,20) %0.88:0.01:0.90   [0.888] % 
    fprintf('Solving for a = %.2f\n', a);

    cvx_begin SDP
        variable P(n,n) semidefinite  
        minimize( -(log_det((P))) )
        subject to
            P - 0.0001 * eye(n) >= 0;
            [a*P - F'*P*F,    -F'*P*G;
             -(F'*P*G)', (1-a)*R - G'*P*G] >= 0;
    cvx_end

    fprintf('Objective value (log_det): %.4f\n', cvx_optval);

    % Save current results
    a_vals(end+1) = a;
    obj_vals(end+1) = cvx_optval;

    % Update best if current is better
    if cvx_optval < best_obj
        fprintf('New best solution found at a = %.2f!\n', a);
        best_obj = cvx_optval;
        best_P = P;
        best_a = a;
    end

    fprintf('\n');
end

% Print summary
fprintf('======== Summary ========\n');
fprintf(' a      |  Objective\n');
fprintf('--------|------------\n');
for i = 1:length(a_vals)
    fprintf(' %.3f  |  %.5f\n', a_vals(i), obj_vals(i));
end

fprintf('\nBest objective value: %.5f at a = %.3f\n', best_obj, best_a);
%disp('Best P matrix:');
%disp(best_P);





%Projection onto x1-x2 plane
x = sdpvar(2 , 1);
 constraints = [(x)'*(best_P([1 2],[1 2]))*(x) <= m;];
 S = YSet(x, constraints);
 S.isBounded()
 figure(1)
 hold on
 S.plot('color', 'lightblue','alpha', 0.2, 'linewidth', 1, 'linestyle', '-')
 figure(1)
 hold on

% Add knowledge on the unsafe states

cvx_begin SDP
    variable Y(n,n) semidefinite
    variable R_hat(m,m) semidefinite diagonal
    minimize( trace(R_hat) )
    subject to
        R_hat >= R;

        for i = 1:num_constraints
            ci = C(:, i);
            ci' * Y * ci <= (b^2) / m;
        end

        [a*Y        zeros(n,m) Y*F';
         zeros(m,n) (1-a)*R_hat G';
         F*Y         G          Y] >= 0;

cvx_end

% % R_hat 
% % inv(Y) 
% % eig(Y)

Ub_controlled = sqrt(1 ./ diag(R_hat));

fprintf('\nPhysical actuator bounds:\n');
for i = 1:length(Ub)
    fprintf('Ub%d = %.4f\n', i, Ub(i));
end

fprintf('\nControlled actuator bounds:\n');
for i = 1:length(Ub_controlled)
    fprintf('Ub%d_controlled = %.4f\n', i, Ub_controlled(i));
end






%Projection onto x1-x2 plane
figure(1)
hold on
x1 = sdpvar(2,1);
constraints1 = [(x1)'*inv(Y([1 2],[1 2]))*x1<=m;];
S1 = YSet(x1, constraints1);
S1.isBounded()
figure(1)
hold on
S1.plot('color', 'red','alpha', 0.2, 'linewidth', 1, 'linestyle', '-')
%%
fig_lims = 10;
D1 = [];
for i = -fig_lims:0.1:fig_lims
    x2 = i;
    x1 = -b;
    D1 = [D1 [x1;x2]];
end
D2 = [];
for j = -fig_lims:0.1:fig_lims
    x1 = j;
    x2 = -b;
    D2 = [D2 [x1;x2]];
end
figure (1)
hold on
box on
plot(D1(1,:),D1(2,:),'r--', 'linewidth', 1.5)
plot(D2(1,:),D2(2,:),'r--', 'linewidth', 1.5)
xlim([-fig_lims fig_lims]);  
ylim([-fig_lims fig_lims]);
xlabel('$\tilde{d}_1$', 'Interpreter', 'Latex')
ylabel('$\tilde{d}_2$', 'Interpreter', 'Latex')
axis square










% simulate trajectory
% Define simulation parameters
extra_braking_time = 200;
t_sim = 90+extra_braking_time;                % time when attack is active but no emergency brake
dt_sim = Dt;               % time step [s]
F_sim = F;                 % system matrix
G_sim = G;                 % input matrix
t_emergency_brake = t_sim-extra_braking_time;
t_attack = 300;


% Initial state: [d1_tilde; d2_tilde; v0_tilde; v1_tilde; v2_tilde]
x_0 = zeros(1,n);     % column vector



sim_runs = 1;
sim_steps = round(t_sim / dt_sim);

% Time vector
time = linspace(0, t_sim, sim_steps);


% simulate trajectory with act bounds
%u_cacc = [-Ub_controlled(1); Ub_controlled(2); Ub_controlled(3)];


% Simulation parameters
use_constant_attack = true;  
use_random_attack = false;
use_sinusoidal_attack = false;


                       % Simulation index

% save data for comparison figure
% Load Python's numpy module
np = py.importlib.import_module('numpy');



if use_constant_attack
    sim_folder = 'simulation_data_statistics_constant_attack/3Kafash et al.';  % Folder to save results
elseif use_random_attack
    sim_folder = 'simulation_data_statistics_random_attack/3Kafash et al.';
elseif use_sinusoidal_attack
    sim_folder = 'simulation_data_statistics_sinusoidal_attack/3Kafash et al.';
end



% save constant simulation parameters
py.numpy.save(py.os.path.join(sim_folder, 'time_vec.npy'), py.numpy.array(time));
py.numpy.save(py.os.path.join(sim_folder, 'time_to_attack.npy'), py.numpy.array(t_attack));
py.numpy.save(py.os.path.join(sim_folder, 'time_to_brake.npy'), py.numpy.array(t_emergency_brake));
py.numpy.save(py.os.path.join(sim_folder, 'use_constant_attack.npy'), py.numpy.array(use_constant_attack));
py.numpy.save(py.os.path.join(sim_folder, 'use_random_attack.npy'), py.numpy.array(use_random_attack));
py.numpy.save(py.os.path.join(sim_folder, 'use_sinusoidal_attack.npy'), py.numpy.array(use_sinusoidal_attack));


% run simulations
add_label = true;
h = waitbar(0, 'Processing Simulation Runs...', 'Name', 'Simulation Progress');
figure(h); 
for s = 0:sim_runs-1

    % Alpha filter setup
    if use_constant_attack
        alpha_filters = zeros(n_follower_vehicles + 1, 1);
    else
        alpha_filters = 0.01 + (1 - 0.01) * rand(n_follower_vehicles + 1, 1);
    end
    
    % Initial attack signal and sinusoidal parameters
    prev_atck_signal = u_min + (u_max - u_min) * rand(n_follower_vehicles + 1, 1);
    initial_phases = 2 * pi * rand(n_follower_vehicles + 1, 1);
    frequencies = 10 * rand(n_follower_vehicles + 1, 1);
    
    % Display parameters
    fprintf('---------------------\n');
    fprintf('Simulation parameters:\n');
    disp('alpha_filters ='); disp(alpha_filters);
    disp('initial_phases ='); disp(initial_phases);
    disp('frequencies ='); disp(frequencies);
    fprintf('---------------------\n');
    
    % simulate trajectory
    [x_history,x_leader_history] = simulate_trajecotry( sim_steps, F_sim, G_sim, x_0,dt_sim,t_attack,t_emergency_brake,v_d,P, ...
                                                        m,alpha_filters, prev_atck_signal,u_min,Ub_controlled, ...
                                                        use_random_attack, use_constant_attack, use_sinusoidal_attack, ...
                                                        initial_phases, frequencies);
    
    


    % Custom plot function (you need to define or adapt this)
    plot(x_history(1, :), x_history(2, :), 'Color', [0.5,0.5,0.5]);
    % Find the step where the emergency brake occurs
    step_brake = round(t_emergency_brake / Dt);  % Compute the corresponding time step
    % Plot a red X when the emergency brake happens
    plot(x_history(1, step_brake), x_history(2, step_brake), 'ko', 'MarkerSize', 10, 'LineWidth', 2);



    % Save to .npy format using numpy
    
    % produce matrices
    v_sim = transpose(x_history(n_follower_vehicles+1:end,:)+v_d);
    d = 6;

    x_sim = zeros(n_follower_vehicles+1,sim_steps);
    x_sim(1,:) = x_leader_history;

    for nn = 2:n_follower_vehicles
        x_sim(nn,:) = x_history(nn-1,:) - d + x_sim(nn-1,:);
    end

    x_sim = transpose(x_sim);



    
    % Save alpha_filters, initial_phases, frequencies using Python's NumPy
    py.numpy.save(py.os.path.join(sim_folder, sprintf('%dalpha_filters.npy', s)), ...
                  py.numpy.array(alpha_filters));
    py.numpy.save(py.os.path.join(sim_folder, sprintf('%dinitial_phases.npy', s)), ...
                  py.numpy.array(initial_phases));
    py.numpy.save(py.os.path.join(sim_folder, sprintf('%dfrequencies.npy', s)), ...
                  py.numpy.array(frequencies));

    py.numpy.save(py.os.path.join(sim_folder, sprintf('%dx_sim.npy', s)), ...
                  py.numpy.array(x_sim));
    py.numpy.save(py.os.path.join(sim_folder, sprintf('%dv_sim.npy', s)), ...
                  py.numpy.array(v_sim));







    % Plot 2D trajectory in d1-d2 space
    if add_label
        label_traj = 'trajectory attack-mitigation ON';
    else
        label_traj = '';
    end

    % Update the progress bar
    waitbar(s / sim_runs, h, sprintf('Processing Run %d of %d...', s, sim_runs));
end
% remove waitbar
close(h);




% Create a new figure
figure(10); clf;

% Define the starting index
start_idx = n_follower_vehicles + 1;

% Number of states to plot
num_states = size(x_history, 1);
colors = lines(num_states);  % Generate distinct colors automatically

hold on;
for i = start_idx:num_states
    if i == start_idx
    plot(time, x_history(i, :) + v_d, '--', 'Color', colors(i, :), 'LineWidth', 1.5);    
    else
    plot(time, x_history(i, :) + v_d, 'Color', colors(i, :), 'LineWidth', 1.5);
    end
end
hold off;

xlabel('Time Step');
ylabel('State Values');
title('States v_i Over Time');

% Generate dynamic legend labels
legend_labels = arrayfun(@(i) sprintf('v_{%d}', i - start_idx), start_idx:num_states, 'UniformOutput', false);
legend(legend_labels, 'Location', 'best');

grid on;


% Create a new figure
figure(11); clf;

% Number of states to plot
colors = lines(num_states);  % Generate distinct colors automatically

hold on;
for i = 1:n_follower_vehicles
    
    if i == 1
    plot(time, x_history(i, :),'--', 'Color', colors(i, :), 'LineWidth', 1.5);   
    else
    plot(time, x_history(i, :), 'Color', colors(i, :), 'LineWidth', 1.5);
    end
end
plot(time, d * ones(1, length(time)), '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1.5);
hold off;

xlabel('Time Step');
ylabel('State Values');
title('States v_i Over Time');

% Generate dynamic legend labels
legend_labels = arrayfun(@(i) sprintf('d_{%d}',1 + i - start_idx), start_idx:num_states, 'UniformOutput', false);
legend(legend_labels, 'Location', 'best');

grid on;














function [F,G,n,m] = build_FG(dt,n_follower_vehicles,beta,kp,kd)
n = 2 * n_follower_vehicles + 1;
m = n_follower_vehicles + 1;

% --- produce F ---


% add ones on diag
F = eye(n,n);

for ii = 1:n_follower_vehicles
% add velocity affecting the position
F(ii,ii + n_follower_vehicles) =  - dt;
F(ii,ii + n_follower_vehicles+1) =  dt;
end

%first and last follower are different
% leader
F(1+n_follower_vehicles,1 + n_follower_vehicles) =  beta-kd + F(1+n_follower_vehicles,1 + n_follower_vehicles);
F(1+n_follower_vehicles,1)   =  +kp;
F(1+n_follower_vehicles,1 + n_follower_vehicles +1)   =  kd;

%last follower
F(1+2*n_follower_vehicles,1 + 2*n_follower_vehicles) =  beta-kd + F(1+2*n_follower_vehicles,1 + 2*n_follower_vehicles);
F(1+2*n_follower_vehicles,n_follower_vehicles) =  -kp;
F(1+2*n_follower_vehicles,1 + 2*n_follower_vehicles -1)   =  kd;

for ii = 2:n_follower_vehicles % followers in the middle 
    % diagonal terms
    F(ii+n_follower_vehicles,ii + n_follower_vehicles) =  beta-2*kd + F(ii+n_follower_vehicles,ii + n_follower_vehicles); 

    % position dependent terms
    F(ii+n_follower_vehicles,ii-1) =  -kp;
    F(ii+n_follower_vehicles,ii)   =  +kp;

    % velocity dependent terms
    F(ii+n_follower_vehicles,ii + n_follower_vehicles -1)   =  kd;
    F(ii+n_follower_vehicles,ii + n_follower_vehicles +1)   =  kd;
end

% --- produce G ---

G = [zeros(n_follower_vehicles,m);
     dt*eye(m)];

end




function [x_history,x_leader_history] = simulate_trajecotry(sim_steps, F_sim, G_sim, x_0,dt,t_attack,t_emergency_brake,v_d,P, ...
                                                            m,alpha_filters, prev_atck_signal,u_min ,Ub_controlled, ...
                                                            use_random_attack, use_constant_attack, use_sinusoidal_attack, ...
                                                            initial_phases, frequencies)
    v_max =  27.78;


    % Preallocate state and input histories
    x_history = zeros(length(x_0), sim_steps);
    u_history = zeros(m, sim_steps);
    x_leader_history = zeros(1,sim_steps);
    ellipse_val = zeros(1, sim_steps);

    % Assign initial condition
    x_history(:, 1) = x_0;

    for t = 2:sim_steps

        if t * dt > t_attack
            [u_atck_vector, prev_atck_signal] = generate_attack_vector( ...
            alpha_filters, prev_atck_signal, Ub_controlled, ...
            use_random_attack, use_constant_attack, use_sinusoidal_attack, ...
            initial_phases, frequencies, t, dt);

            u_cacc = u_atck_vector;

        else
            u_cacc = zeros(m,1);
        end


        % State update
        candidate_new_x = F_sim * x_history(:, t-1) + G_sim * u_cacc;

        % Define control input (example: emergency brake)
        if t * dt > t_emergency_brake
            candidate_new_x(m) = x_history(m, t-1) + u_min * dt; %maximum braking capacity
        end


        % apply velocity constraints
        for jj = m:length(candidate_new_x)
            % Apply velocity constraints
            if candidate_new_x(jj) + v_d < 0
                candidate_new_x(jj) = -v_d;
            elseif candidate_new_x(jj) + v_d > v_max
                candidate_new_x(jj) = v_max - v_d;
            end
        end


        x_history(:, t) = candidate_new_x;
        x_leader_history(t) = x_leader_history(t-1) + (candidate_new_x(m)+v_d) * dt;

        % Store control input
        u_history(:, t) = u_cacc;

        % Evaluate quadratic form (like -log(det(P))) for the ellipse
        ellipse_val(t) = x_history(:, t-1)' * P * x_history(:, t-1);
    end
end





function [u_atck_vector, prev_atck_signal] = generate_attack_vector( ...
    alpha_filters, prev_atck_signal, Ub_controlled, ...
    use_random_attack, use_constant_attack, use_sinusoidal_attack, ...
    initial_phases, frequencies, t, dt_int)

    % Number of attack channels (i.e., number of cars)
    N = length(alpha_filters);
    u_atck_vector = zeros(N, 1);
    % Generate random base signal and update history for each car
    random_noise = rand(N, 1);
    u_atck_random = zeros(N, 1);

    for kk = 1:N
        random_noise(kk) = -Ub_controlled(kk) + 2 * Ub_controlled(kk) * random_noise(kk);
        u_atck_random(kk) = (1 - alpha_filters(kk)) * prev_atck_signal(kk) + ...
                            alpha_filters(kk) * random_noise(kk);
        prev_atck_signal(kk) = u_atck_random(kk);  % Update for next iteration
    end

   
    % Determine base attack
    if use_random_attack || use_constant_attack
        u_atck_vector = u_atck_random;
    end

    % Add sinusoidal components if needed for each car
    if use_sinusoidal_attack
        for kk = 1:N
            sinusoidal = Ub_controlled(kk) * sin(initial_phases(kk) + (t * dt_int) ./ frequencies(kk) * 2 * pi);
            u_atck_vector(kk) = u_atck_vector(kk) + sinusoidal;
        end

    end

    % Saturation (clipping the values within the bounds for each car)
    u_atck_vector = min(max(u_atck_vector, -Ub_controlled), Ub_controlled);
end


