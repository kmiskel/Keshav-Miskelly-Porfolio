close all;
% Simulation Parameters
m = 10;             % Mass (kg)        
dt = 0.01;          % Time step (s)
t_final = 500;      % Final time (s)
num_steps = t_final / dt;
orientation = 0;
Diameter = 5;
F_max = 30;

%range of parameters to test
% b_values = 0.1:0.5:5;
% F_max_values = 1:5:50;

%first i need to orient the plane 180 deg of its current position

%im going to call my a single point has (orientation, x, y)
%
%orientation has a metric 

targets = [ pi/2, 2.5, Diameter/2; pi, 0, Diameter; pi, -304.8, Diameter; 0, -304.8, 0; 0, 0, 0];
target_index = 1;

% Initialize arrays
time = 0:dt:t_final;
x = zeros(1, num_steps + 1);  % x-position array
y = zeros(1, num_steps + 1);  % y-position array
vx = zeros(1, num_steps + 1); % x-velocity array
vy = zeros(1, num_steps + 1); % y-velocity array
Fx = zeros(1, num_steps + 1); % x-force array
Fy = zeros(1, num_steps + 1); % y-force array
theta = zeros(1, num_steps + 1); %angle position array

%vx and vy and Fx and Fy will not be able to be handled as discretely in
%actual

% Initial conditions
x(1) = 0;    % Initial x-position (m)
theta(1) = 0;
y(1) = 0;    % Initial y-position (m)
vx(1) = 50;  % Initial x-velocity (m/s)
vy(1) = 0;   % Initial y-velocity (m/s)

%control loop for 2D movement

% Control loop for 2D movement with smooth path control
for k = 1:num_steps
    target_theta = targets(target_index, 1);
    target_x = targets(target_index, 2);
    target_y = targets(target_index, 3);

    % Compute differences
    dtheta = target_theta - theta(k);
    dx = target_x - x(k);
    dy = target_y - y(k);
    distance = sqrt(dx^2 + dy^2);

    % Check if near the target
    if distance < 0.1
        if target_index < size(targets, 1)
            target_index = target_index + 1; % Move to the next target
        else
            break; % Stop if all targets are reached
        end
    end

    % Desired unit vector toward the target
    if distance > 0
        ux = dx / distance;
        uy = dy / distance;
    else
        ux = 0;
        uy = 0;
    end

    % Desired velocity vector (pass-through target with smooth transition)
    speed_desired = 50; % Desired speed (m/s)
    vx_target = speed_desired * ux;
    vy_target = uy;

    Kp = 400; % Scales up with distance


    % Proportional control for velocity alignment
    % Proportional gain, this is best around 100-400 for a little costrained system
    Fx(k) = Kp * (vx_target - vx(k));
    Fy(k) = Kp * (vy_target - vy(k));

    F_magnitude = sqrt(Fx(k)^2 + Fy(k)^2);

    if F_magnitude > F_max
        speed_desired = speed_desired * (F_max / F_magnitude);
    end

    % Update positions, velocities, and orientation
    vx(k + 1) = vx(k) + Fx(k) / m * dt;
    vy(k + 1) = vy(k) + Fy(k) / m * dt;

    x(k + 1) = x(k) + vx(k) * dt;
    y(k + 1) = y(k) + vy(k) * dt;

    % Update orientation based on velocity direction
    if vx(k + 1) ~= 0 || vy(k + 1) ~= 0
        theta(k + 1) = atan2(vy(k + 1), vx(k + 1));
    else
        theta(k + 1) = theta(k); % Maintain previous orientation if no motion
    end
end

% Plot results
figure;
subplot(2, 1, 1);
plot(x(1:k + 1), y(1:k + 1), '-o');
hold on;
plot(targets(:, 2), targets(:, 3), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('x-position (m)');
ylabel('y-position (m)');
title('Improved 2D Time-Optimal Path');
legend('Path', 'Targets');
grid on;

subplot(2, 1, 2);
plot(time(1:k + 1), sqrt(Fx(1:k + 1).^2 + Fy(1:k + 1).^2));
xlabel('Time (s)');
ylabel('Force Magnitude (N)');
title('Force Magnitude vs. Time');
grid on;

figure;
plot(time(1:k+1),sqrt(vx(1:k+1).^2 + vy(1:k+1).^2));
xlabel('Time (s)'); ylabel('Velocity-x');
optimal_time = time(k+1);
disp(optimal_time);

