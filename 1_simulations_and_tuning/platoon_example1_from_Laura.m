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
F = [1  0 -Dt           Dt           0;
     0  1  0           -Dt           Dt;
     kp 0  (1+beta)-kd  kd           0;
    -kp kp kd          (1+beta)-2*kd kd;
     0 -kp 0            kd           (1+beta)-kd];
G = [zeros(2,3);
     Dt*eye(3)];
n = 5;
m = 3;

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

% Ub1 = 1.1;
% Lb1 = -1.1;
% 
% Ub2 = 0.9;
% Lb2 = -0.9;
% 
% Ub3 = 1.05;
% Lb3 = -1.05;
u_max_actuators = 4.905; % this is the real max acc capacity
v_d = 25; % desired platooning velocity

u_max = 7.848; 


Ub1 = u_max;
Lb1 = -u_max;

Ub2 = u_max;
Lb2 = -u_max;

Ub3 = u_max;
Lb3 = -u_max;


gamma1 = Ub1^2; %sqrt(Ub1);
gamma2 = Ub2^2; %sqrt(Ub2);
gamma3 = Ub3^2; %sqrt(Ub3);

Ub = [Ub1; Ub2; Ub3];
Lb = [Lb1; Lb2; Lb3];


R = zeros (m,m);
R(1,1) = 1/gamma1;
R(2,2) = 1/gamma2;
R(3,3) = 1/gamma3;

% Dangerous regions: -x1>=d1* and -x2>=d2*
% We need to rewrite them in the form ci'x =bi
c1 = zeros(n,1);
c1(1,1) = 1;
c2 = zeros(n,1);
c2(2,1) = 1;
b1 = -6;
b2 = -6;

%a = 0.86;
% Initialize best objective value and solution
best_obj = Inf;
best_P = [];
best_a = NaN;

% Initialize arrays to store results
a_vals = [];
obj_vals = [];

for a = [0.888] %linspace(0.88,0.90,100) %0.88:0.01:0.90
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
disp('Best P matrix:');
disp(best_P);





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
        c1'*Y*c1 <=(b1^2)/m;
        c2'*Y*c2 <=(b1^2)/m;
        
        [a*Y        zeros(n,m) Y*F';
         zeros(m,n) (1-a)*R_hat G';
         F*Y         G          Y]>=0;
        
cvx_end
R_hat 
inv(Y) 
eig(Y)

% new bounds
Ub1_controlled = sqrt((1/R_hat(1,1)));
Ub2_controlled = sqrt((1/R_hat(2,2)));
Ub3_controlled = sqrt((1/R_hat(3,3)));






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
    x1 = -b1;
    D1 = [D1 [x1;x2]];
end
D2 = [];
for j = -fig_lims:0.1:fig_lims
    x1 = j;
    x2 = -b2;
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







%evaluate reachable set using CORA
C = [1 0 0 0 0;
     0 1 0 0 0];
% Define system
sys = linearSysDT(F, G, [],C,Dt);

% initial set upper and lower bound
R0_ub = 0.1 * [2;2;1;1;1];
R0_lb = -0.1* [2;2;1;1;1];
% U_ub = [u_max_actuators;u_max_actuators;u_max_actuators;];
% U_lb = [-u_max;-u_max;-u_max;];
U_ub = [Ub1_controlled;Ub2_controlled;Ub3_controlled;];
U_lb = [-Ub1_controlled;-Ub2_controlled;-Ub3_controlled;];


% parameter
params.tFinal = 100;
params.R0 = zonotope(interval(R0_lb,R0_ub));
params.U = zonotope(interval(U_lb,U_ub));
% reachability settings
options.timeStep = Dt;
options.zonotopeOrder = 40;
options.taylorTerms = 2;
% reachability analysis
R = reach(sys,params,options);

figure(1); hold on;
 
%useCORAcolors("CORA:contDynamics")
plot(R,[1,2],'DisplayName','Reachable set','FaceAlpha',0.2)
%plot(R.R0,[1,2],'DisplayName','Initial set')









% simulate trajectory
% Define simulation parameters
t_sim = 21;                % total simulation time [s]
dt_sim = Dt;               % time step [s]
F_sim = F;                 % system matrix
G_sim = G;                 % input matrix
t_emergency_brake = 11;
t_attack = 1;


% Initial state: [d1_tilde; d2_tilde; v0_tilde; v1_tilde; v2_tilde]
x_0 = [0; 0; 0; 0; 0];     % column vector

sim_runs = 1;
sim_steps = round(t_sim / dt_sim);

% Time vector
time = linspace(0, t_sim, sim_steps);
add_label = true;
for run_idx = 1:sim_runs
    % Preallocate state and input histories
    x_history = zeros(length(x_0), sim_steps);
    u_history = zeros(3, sim_steps);
    ellipse_val = zeros(1, sim_steps);

    % Assign initial condition
    x_history(:, 1) = x_0;

    for i = 2:sim_steps
        % Define control input 
        if i * dt_sim > t_attack
            u_cacc = [Lb1; u_max_actuators; u_max_actuators];
        else
            u_cacc = [0; 0; 0];
        end


        % State update
        x_history(:, i) = F_sim * x_history(:, i-1) + G_sim * u_cacc;

        % Store control input
        u_history(:, i) = u_cacc;

        % Evaluate quadratic form (like -log(det(P))) for the ellipse
        ellipse_val(i) = x_history(:, i-1)' * best_P * x_history(:, i-1);
    end

    % Plot 2D trajectory in d1-d2 space
    if add_label
        label_traj = 'trajectory attack-mitigation OFF';
    else
        label_traj = '';
    end

    % Custom plot function (you need to define or adapt this)
%     plot(x_history(1, :), x_history(2, :), 'DisplayName', label_traj, 'Color', 'blue');

end





% simulate trajectory with act bounds
u_cacc = [-Ub1_controlled; Ub2_controlled; Ub3_controlled];

fprintf('\n Physical actuator bounds:\n');
disp(Ub1);
disp(Ub2);
disp(Ub3);

fprintf('\n controlled actuator bounds:\n');
disp(Ub1_controlled);
disp(Ub2_controlled);
disp(Ub3_controlled);


add_label = true;
for run_idx = 1:sim_runs

    x_history = simulate_trajecotry(sim_steps, F_sim, G_sim, x_0,dt_sim,t_attack,t_emergency_brake,u_cacc,v_d,best_P);

    % Plot 2D trajectory in d1-d2 space
    if add_label
        label_traj = 'trajectory attack-mitigation ON';
    else
        label_traj = '';
    end

    % Custom plot function (you need to define or adapt this)
    plot(x_history(1, :), x_history(2, :), 'DisplayName', label_traj, 'Color', 'red');
    % Find the step where the emergency brake occurs
    step_brake = round(t_emergency_brake / Dt);  % Compute the corresponding time step
    % Plot a red X when the emergency brake happens
    plot(x_history(1, step_brake), x_history(2, step_brake), 'ro', 'MarkerSize', 10, 'LineWidth', 2);


end


% Create a new figure
figure(10);
plot(time, x_history(3, :) + v_d, 'r', ...
     time, x_history(4, :) + v_d, 'g', ...
     time, x_history(5, :) + v_d, 'b');
xlabel('Time Step');
ylabel('State Values');
title('States v_0, v_1, v_2 Over Time');
legend('v_0','v_1','v_2');
grid on;







function x_history = simulate_trajecotry(sim_steps, F_sim, G_sim, x_0,dt,t_attack,t_emergency_brake,u_cacc_atk,v_d,P)
    v_max =  27.78;


    % Preallocate state and input histories
    x_history = zeros(length(x_0), sim_steps);
    u_history = zeros(3, sim_steps);
    ellipse_val = zeros(1, sim_steps);

    % Assign initial condition
    x_history(:, 1) = x_0;

    for i = 2:sim_steps
        if i * dt > t_attack
            u_cacc = u_cacc_atk;
        else
            u_cacc = [0; 0; 0];
        end


        % State update
        candidate_new_x = F_sim * x_history(:, i-1) + G_sim * u_cacc;

        % Define control input (example: emergency brake)
        if i * dt > t_emergency_brake
            candidate_new_x(3) = x_history(3, i-1) -7.848 * dt; %maximum braking capacity
        end


        % apply velocity constraints
        if candidate_new_x(3) + v_d < 0
            candidate_new_x(3) = -v_d;
        elseif candidate_new_x(3) + v_d > v_max
            candidate_new_x(3) = v_max-v_d;
        end
        if candidate_new_x(4) + v_d < 0
            candidate_new_x(4) = -v_d;
        elseif candidate_new_x(4) + v_d > v_max
            candidate_new_x(4) = v_max-v_d;
        end
        if candidate_new_x(5) + v_d < 0
            candidate_new_x(5) = -v_d;
        elseif candidate_new_x(5) + v_d > v_max
            candidate_new_x(5) = v_max-v_d;
        end

        x_history(:, i) = candidate_new_x;

        % Store control input
        u_history(:, i) = u_cacc;

        % Evaluate quadratic form (like -log(det(P))) for the ellipse
        ellipse_val(i) = x_history(:, i-1)' * P * x_history(:, i-1);
    end
end
