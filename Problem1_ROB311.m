%Problem 1 _ 16/10/2017 _ Raul Da Souza, Thomas Gauthier, Nicolas Gauthier%

% Initilization

% utility functions
U = zeros(1,6);
U1 = zeros(1,6);
% discount factor for the value iteration
gamma = 0.999;
%gamma = 0.1;
% stopping criterion for the value iteration
epsilon = 0.01;
% Rewards for states
R = [-0.1, -0.1, 1, -0.1, -0.1, -0.05];
% Possible actions
actions = {'north', 'south', 'east', 'west', 'still'};

% Transition models
T_north =  [0.05 0.05 0 0.9 0 0;
            0.05 0 0.05 0 0.9 0;
            0 0.05 0.05 0 0 0.9;
            0 0 0 0.95 0.05 0;
            0 0 0 0.05 0.9 0.05;
            0 0 0 0 0.05 0.95];

T_south =  [0.95 0.05 0 0 0 0;
            0.05 0.9 0.05 0 0 0;
            0 0.05 0.95 0 0 0;
            0.9 0 0 0.05 0.05 0;
            0 0.9 0 0.05 0 0.05;
            0 0 0.9 0 0.05 0.05];

T_east =   [0.05 0.9 0 0.05 0 0;
            0 0.05 0.9 0 0.05 0;
            0 0 0.95 0 0 0.05;
            0.05 0 0 0.05 0.9 0;
            0 0.05 0 0 0.05 0.9;
            0 0 0.05 0 0 0.95];
        
T_west =   [0.95 0 0 0.05 0 0;
            0.9 0.05 0 0 0.05 0;
            0 0.9 0.05 0 0 0.05;
            0.05 0 0 0.95 0 0;
            0 0.05 0 0.9 0.05 0;
            0 0 0.05 0 0.9 0.05];
        
T_still = eye(6);

% Initialise the RMSE greater than the stopping criterion
RMSE = 1+ epsilon;
% Number of iterations
nb_iterations = 0;
while RMSE > epsilon
    nb_iterations = nb_iterations +1;
    U1 = U;
    for i = 1:6
        U(i) = R(i) + gamma*max([T_north(i,:)*U1', T_south(i,:)*U1', T_east(i,:)*U1', T_west(i,:)*U1', T_still(i,:)*U1']);
    end
    RMSE = sqrt(sum((U-U1).^2))/6;
end

% Compute the policy from the transition models and utility function
policy = {};
for i = 1:6
    [~,action] = max([T_north(i,:)*U1', T_south(i,:)*U1', T_east(i,:)*U1', T_west(i,:)*U1', T_still(i,:)*U1']);
    policy(i) = actions(action);
end