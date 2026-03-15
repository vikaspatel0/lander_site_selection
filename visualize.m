% visualize_trajectory.m
clc; clear; %close all;

% Load the trajectory data
data = load('lander_trajectory_reachable.mat');
grid_mean = data.mean_history;
std_history = data.std_history;
lander_i = data.lander_i;
lander_j = data.lander_j;
lander_z = data.lander_z;
actions = data.actions;
rewards = data.rewards;
n_steps = data.n_steps;

% Initialize figure
fig = figure('Name', 'Lander Trajectory Viewer', 'Position', [100, 100, 1400, 500]);

% Create data structure to pass to callbacks
handles = struct();
handles.current_step = 1;
handles.grid_mean = grid_mean;
handles.std_history = std_history;
handles.lander_i = lander_i;
handles.lander_j = lander_j;
handles.lander_z = lander_z;
handles.actions = actions;
handles.rewards = rewards;
handles.n_steps = n_steps;

% Create navigation buttons
handles.btn_prev = uicontrol('Style', 'pushbutton', 'String', '← Previous', ...
    'Position', [20, 20, 100, 30], 'FontSize', 10);
handles.btn_next = uicontrol('Style', 'pushbutton', 'String', 'Next →', ...
    'Position', [130, 20, 100, 30], 'FontSize', 10);

% Step counter text
handles.step_text = uicontrol('Style', 'text', 'String', '', ...
    'Position', [250, 20, 200, 30], 'FontSize', 10, 'FontWeight', 'bold');

% Store handles in figure
guidata(fig, handles);

% Set button callbacks
set(handles.btn_prev, 'Callback', {@prev_step, fig});
set(handles.btn_next, 'Callback', {@next_step, fig});

% Plot initial state
update_plot(fig);

%exportgraphics(gcf, "initial_test_fig.png", "Resolution", 600)

function update_plot(fig)
    handles = guidata(fig);
    current_step = handles.current_step;
    
    % Clear previous plots (but keep buttons)
    subplot(1, 3, 1);
    cla;
    
    % Terrain map
    imagesc(handles.grid_mean(:, :, current_step));
    colorbar;
    title('Terrain (Mean Reward)', 'FontSize', 11);
    xlabel('Column (j)');
    ylabel('Row (i)');
    axis equal tight;
    colormap(gca, 'parula');
    xlim([13 17])
    ylim([3 7])

    % Plot lander position
    hold on;
    plot(handles.lander_j(current_step), handles.lander_i(current_step), 'rp', ...
        'MarkerSize', 20, 'MarkerFaceColor', 'r', 'LineWidth', 2);
    
    % Plot trajectory up to current step
    if current_step > 1
        plot(handles.lander_j(1:current_step), handles.lander_i(1:current_step), ...
            'w--', 'LineWidth', 2);
        plot(handles.lander_j(1:current_step), handles.lander_i(1:current_step), ...
            'wo', 'MarkerSize', 4, 'MarkerFaceColor', 'w');
    end
    hold off;
    
    % Standard deviation map
    subplot(1, 3, 2);
    cla;
    
    imagesc(handles.std_history(:, :, current_step));
    colorbar;
    title('Uncertainty (Std Dev)', 'FontSize', 11);
    xlabel('Column (j)');
    ylabel('Row (i)');
    axis equal tight;
    colormap(gca, 'hot');
    caxis([0, max(handles.std_history(:))]);
    
    % Plot lander position
    hold on;
    plot(handles.lander_j(current_step), handles.lander_i(current_step), 'cp', ...
        'MarkerSize', 5, 'MarkerFaceColor', 'r', 'LineWidth', 2);

    hold off;
    
    % Risk-sensitive reward (10th percentile)
    subplot(1, 3, 3);
    cla;
    
    risk_percentile = 0.1;
    risk_reward = handles.grid_mean(:, :, current_step) + norminv(risk_percentile) * handles.std_history(:, :, current_step);
    imagesc(risk_reward);
    colorbar;
    title('Risk-Sensitive Reward (10th percentile)', 'FontSize', 11);
    xlabel('Column (j)');
    ylabel('Row (i)');
    axis equal tight;
    colormap(gca, 'jet');
    
    % Plot lander position
    hold on;
    plot(handles.lander_j(current_step), handles.lander_i(current_step), 'mp', ...
        'MarkerSize', 20, 'MarkerFaceColor', 'm', 'LineWidth', 2);
    if current_step > 1
        plot(handles.lander_j(1:current_step), handles.lander_i(1:current_step), ...
            'w--', 'LineWidth', 2);
        plot(handles.lander_j(1:current_step), handles.lander_i(1:current_step), ...
            'wo', 'MarkerSize', 4, 'MarkerFaceColor', 'w');
    end
    hold off;
    
    % Update title with step info
    if current_step <= length(handles.actions)
        action_str = handles.actions{current_step};
        reward_str = sprintf('%.4f', handles.rewards(current_step));
    else
        action_str = 'LANDED';
        reward_str = 'N/A';
    end
    
    sgtitle(sprintf('Step %d/%d | Position: (%d,%d) | Altitude: %d | Action: %s | Reward: %s', ...
        current_step, handles.n_steps, ...
        handles.lander_i(current_step), handles.lander_j(current_step), ...
        handles.lander_z(current_step), action_str, reward_str), ...
        'FontSize', 13, 'FontWeight', 'bold');
    
    % Update button states
    if current_step == 1
        set(handles.btn_prev, 'Enable', 'off');
    else
        set(handles.btn_prev, 'Enable', 'on');
    end
    
    if current_step == handles.n_steps
        set(handles.btn_next, 'Enable', 'off');
    else
        set(handles.btn_next, 'Enable', 'on');
    end
    
    % Update step counter
    set(handles.step_text, 'String', sprintf('Step %d of %d', current_step, handles.n_steps));
    
    drawnow;
end

function prev_step(~, ~, fig)
    handles = guidata(fig);
    if handles.current_step > 1
        handles.current_step = handles.current_step - 1;
        guidata(fig, handles);
        update_plot(fig);
    end
end

function next_step(~, ~, fig)
    handles = guidata(fig);
    if handles.current_step < handles.n_steps
        handles.current_step = handles.current_step + 1;
        guidata(fig, handles);
        update_plot(fig);
    end
end