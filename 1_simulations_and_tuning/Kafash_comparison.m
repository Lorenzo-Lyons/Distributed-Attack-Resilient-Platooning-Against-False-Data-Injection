clc
close all
clear all

dt = 0.1;
k =  2.456808030112924;
c =  8.69580370835076;
h =  0.11222444889779559;
d =  6.0;



% vehicle limits
u_min_original =  -7.848;
u_max_original =  4.905;
u_max = 7.848; % taking maximum actuation as the largest between braking and accelerating as a conservative emeasure



% build state transition matrices
[F, G] = build_state_transition_matrices(dt, k, c, h);




n = 5;
m = 3;
% Upper and lower bounds on the secondary control action wi
Ub1 = u_max;
Lb1 = -u_max;
gamma1 = Ub1^2;

Ub2 = u_max;
Lb2 = -u_max;
gamma2 = Ub2^2;

Ub3 = u_max;
Lb3 = -u_max;
gamma3 = Ub3^2;

Ub = [Ub1; Ub2; Ub3];
Lb = [Lb1; Lb2; Lb3];


R = zeros (m,m);
R(1,1) = 1/gamma1;
R(2,2) = 1/gamma2;
R(3,3) = 1/gamma3;

% Dangerous regions: -x1>=d1* and -x2>=d2*
% We need to rewrite them in the form ci'x =bi
c1 = zeros(n,1);
c1(1,1) = -1;
c2 = zeros(n,1);
c2(2,1) = -1;
b1 = 1;
b2 = 1;

% a = 0.93;
% Initialize best objective value and solution
best_obj = Inf;
best_P = [];
best_a = NaN;
%best_a = 0.96



% Initialize arrays to store results
a_vals = [];
obj_vals = [];

for a = 0.01:0.1:0.99
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
 constraints = [(x)'*(P([1 2],[1 2]))*(x) <= m;];
 S = YSet(x, constraints);
 S.isBounded()
 figure(1)
 hold on
 S.plot('color','lightblue','linewidth', 1, 'linestyle', '-') %lightblue
 figure(1)
 axis equal
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

%Projection onto x1-x2 plane
figure(1)
hold on
x1 = sdpvar(2,1);
constraints1 = [(x1)'*inv(Y([1 2],[1 2]))*x1<=m;];
S1 = YSet(x1, constraints1);
S1.isBounded()
figure(1)
hold on
S1.plot( 'linewidth', 1, 'linestyle', '-') %'color', 'red'
%%
D1 = [];
for i = -6:0.1:6
    x2 = i;
    x1 = -1;
    D1 = [D1 [x1;x2]];
end
D2 = [];
for j = -6:0.1:6
    x1 = j;
    x2 = -1;
    D2 = [D2 [x1;x2]];
end
figure (1)
hold on
box on
plot(D1(1,:),D1(2,:),'r--', 'linewidth', 1.5)
plot(D2(1,:),D2(2,:),'r--', 'linewidth', 1.5)
xlim([-6 6]);  
ylim([-6 6]);
xlabel('$\tilde{d}_1$', 'Interpreter', 'Latex')
ylabel('$\tilde{d}_2$', 'Interpreter', 'Latex')
axis square











function [F, G] = build_state_transition_matrices(dt, k, c, h)
    % Build the F matrix
    F = eye(5) + [ 0    , 0    , -dt           , +dt           , 0;
                   0    , 0    , 0             , -dt           , +dt;
                   0    , 0    , -k*h*dt - c*dt, 0             , 0;
                  -k*dt , 0    , +c*dt         , -k*h*dt - c*dt, 0;
                   0    , -k*dt, 0             , +c*dt         , -k*h*dt - c*dt ];
    
    % Build the G matrix
    G = [zeros(2,3);
         dt * eye(3)];
    
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

    % test controllability
    test_controllability(F, G)


end


function is_controllable = test_controllability(F, G)
    % Build the controllability matrix
    n = size(F, 1); % number of states
    controllability_matrix = G;
    for i = 1:n-1
        controllability_matrix = [controllability_matrix, F^i * G];
    end

    % Check rank
    rank_C = rank(controllability_matrix);

    % Display result
    fprintf('Rank of controllability matrix: %d (state dimension: %d)\n', rank_C, n);
    if rank_C == n
        disp('System (F,G) is controllable ✅');
        is_controllable = true;
    else
        disp('System (F,G) is NOT controllable ❌');
        is_controllable = false;
    end
end



