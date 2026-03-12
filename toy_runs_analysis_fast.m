clc; clear; close all;

%strats = ["const_p10", "const_p20", "const_p30", "const_p40", "const_p50", "const_p60", "const_p70", "const_p80", "const_p90", "max_perc_beta0_5", "min_perc_beta0_5", "mean_perc_beta0_5", "prev_perc_beta0_5"];
strats = ["const_p10_test0", "const_p20_test0", "const_p30_test0", "const_p40_test0", "const_p50_test0", "const_p60_test0", "const_p70_test0", "const_p80_test0", "const_p90_test0", "max_perc_beta0_5_test0", "min_perc_beta0_5_test0", "mean_perc_beta0_5_test0",...
          "CVaR_alpha10_test0", "CVaR_alpha30_test0", "CVaR_alpha50_test0"];%, "thompson_test0"];
%strats = ["const_p90", "const_p90_test", "const_p90_test0", "const_p90_test1", "mean_perc_beta0_5"];
%strats_names = ["10^{th} %", "20^{th} %", "30^{th} %", "40^{th} %", "50^{th} %", "60^{th} %", "70^{th} %", "80^{th} %", "90^{th} %", "Max \sigma", "Min \sigma", "Mean \sigma", "Previous \sigma"];
strats_names = ["10^{th} %0", "20^{th} %0", "30^{th} %0", "40^{th} %0", "50^{th} %0", "60^{th} %0", "70^{th} %0", "80^{th} %0", "90^{th} %0", "Max \sigma 0", "Min \sigma 0", "Mean \sigma 0",...
                "CVaR \alpha 10", "CVaR \alpha 30", "CVaR \alpha 50"];%, "Thompson"];
%strats_names = ["90^{th}", "90^{th} ts", "90^{th} ts0", "90^{th} ts1", "Mean \sigma"];
reward_transition = ["k0", "k2", "k4", "k7", "k15"];
reward_transition_names = ["k=0", "k=2", "k=4", "k=7", "k=15"];
sigma = ["0_5", "1_0", "2_0", "3_0", "5_0"];
sigma_names = ["0.5", "1", "2", "3", "5"];

% -----------------------------
% 0) PICK ONE METRIC
% -----------------------------
metric_name = "Entropic";   % "Mean","Min","Q05","CVaR10","Entropic","PowerPm3"

alpha = 0.10;   % CVaR tail fraction
q     = 0.05;   % quantile level
beta  = 0.5;    % entropic strength
p     = -3;     % power mean exponent

% Do we need global normalization?
needs_norm = ismember(metric_name, ["PowerPm3"]);  % add others here if needed

% Define metric function (can close over norm01 if needed)
switch metric_name
    case "Mean"
        metric = @(v) mean(v,'omitnan');
    case "Min"
        metric = @(v) min(v);
    case "Q05"
        metric = @(v) quantile(v, q);
    case "CVaR10"
        metric = @(v) mean(v(v <= quantile(v, alpha)), 'omitnan');
    case "Entropic"
        % as in your current code: no normalization
        metric = @(v) -(1/beta) * log(mean(exp(-beta * v), 'omitnan') + eps);
    case "PowerPm3"
        metric = @(v) ( mean( (norm01(v)+eps).^p, 'omitnan' ) ).^(1/p);
    otherwise
        error("Unknown metric_name: %s", metric_name);
end

% -----------------------------
% 2) PASS 2 (or only pass): COMPUTE SCORES ONLY
% -----------------------------
n = numel(strats) * numel(reward_transition) * numel(sigma);
Results = table( ...
    strings(n,1), strings(n,1), strings(n,1), nan(n,1), nan(n,1), nan(n,1), nan(n,1), ...
    'VariableNames', {'strat','reward_transition','sigma','Score', 'perc_5th', 'perc_10th', 'perc_25th'} );

idx = 0;
for i = strats
    for j = reward_transition
        for k = sigma
            idx = idx + 1;

            filename = sprintf("data/batch_%s_%s_sigma%s.csv", i, j, k);
            T = readtable(filename);
            v = T{:,8};
            v = v(:);
            v = v(~isnan(v));

            Results.strat(idx) = i;
            Results.reward_transition(idx) = j;
            Results.sigma(idx) = k;

            if isempty(v)
                Results.Score(idx) = NaN;
            else
                Results.Score(idx) = metric(v);
            end

            Results.perc_5th(idx) = prctile(v, 5);
            Results.perc_10th(idx) = prctile(v, 10);
            Results.perc_25th(idx) = prctile(v, 25);
        end
    end
end

% -----------------------------
% 3) ONE FIGURE WITH 5 SUBPLOTS (one per sigma)
% -----------------------------
fig = figure('Color','w');
t = tiledlayout(fig, 1, 5, 'TileSpacing','compact', 'Padding','compact');
title(t, sprintf("Scores - Metric: %s", metric_name), 'Interpreter','none');

for kk = 1:numel(sigma)
    s = sigma(kk);
    sub = Results(Results.sigma == s, :);

    M = nan(numel(strats), numel(reward_transition));
    for a = 1:numel(strats)
        for b = 1:numel(reward_transition)
            row = sub(sub.strat==strats(a) & sub.reward_transition==reward_transition(b), :);
            if ~isempty(row)
                M(a,b) = row.Score(1);
            end
        end
    end

    ax = nexttile(t, kk);
    imagesc(ax, M);
    axis(ax, 'tight');
    colormap(ax, parula);

    set(ax, ...
        'XTick', 1:numel(reward_transition), 'XTickLabel', reward_transition_names, ...
        'YTick', 1:numel(strats),           'YTickLabel', strats_names, ...
        'TickLabelInterpreter','tex');

    xlabel(ax, 'Reward Transition');
    ylabel(ax, 'Strategy');
    title(ax, sprintf("\\sigma = %s", sigma_names(kk)), 'Interpreter','tex');

    for r = 1:size(M,1)
        for c = 1:size(M,2)
            if ~isnan(M(r,c))
                text(ax, c, r, sprintf('%.3g', M(r,c)), ...
                    'HorizontalAlignment','center', 'Color','w', 'FontSize',8);
            end
        end
    end
end

fig = figure('Color','w');
t = tiledlayout(fig, 1, 5, 'TileSpacing','compact', 'Padding','compact');
title(t, sprintf("5th %%ile - Metric: %s", metric_name), 'Interpreter','none');

for kk = 1:numel(sigma)
    s = sigma(kk);
    sub = Results(Results.sigma == s, :);

    M = nan(numel(strats), numel(reward_transition));
    for a = 1:numel(strats)
        for b = 1:numel(reward_transition)
            row = sub(sub.strat==strats(a) & sub.reward_transition==reward_transition(b), :);
            if ~isempty(row)
                M(a,b) = row.perc_5th(1);
            end
        end
    end

    ax = nexttile(t, kk);
    imagesc(ax, M);
    axis(ax, 'tight');
    colormap(ax, parula);

    set(ax, ...
        'XTick', 1:numel(reward_transition), 'XTickLabel', reward_transition_names, ...
        'YTick', 1:numel(strats),           'YTickLabel', strats_names, ...
        'TickLabelInterpreter','tex');

    xlabel(ax, 'Reward Transition');
    ylabel(ax, 'Strategy');
    title(ax, sprintf("\\sigma = %s", sigma_names(kk)), 'Interpreter','tex');

    for r = 1:size(M,1)
        for c = 1:size(M,2)
            if ~isnan(M(r,c))
                text(ax, c, r, sprintf('%.3g', M(r,c)), ...
                    'HorizontalAlignment','center', 'Color','w', 'FontSize',8);
            end
        end
    end
end

fig = figure('Color','w');
t = tiledlayout(fig, 1, 5, 'TileSpacing','compact', 'Padding','compact');
title(t, sprintf("10th %%ile - Metric: %s", metric_name), 'Interpreter','none');

for kk = 1:numel(sigma)
    s = sigma(kk);
    sub = Results(Results.sigma == s, :);

    M = nan(numel(strats), numel(reward_transition));
    for a = 1:numel(strats)
        for b = 1:numel(reward_transition)
            row = sub(sub.strat==strats(a) & sub.reward_transition==reward_transition(b), :);
            if ~isempty(row)
                M(a,b) = row.perc_10th(1);
            end
        end
    end

    ax = nexttile(t, kk);
    imagesc(ax, M);
    axis(ax, 'tight');
    colormap(ax, parula);

    set(ax, ...
        'XTick', 1:numel(reward_transition), 'XTickLabel', reward_transition_names, ...
        'YTick', 1:numel(strats),           'YTickLabel', strats_names, ...
        'TickLabelInterpreter','tex');

    xlabel(ax, 'Reward Transition');
    ylabel(ax, 'Strategy');
    title(ax, sprintf("\\sigma = %s", sigma_names(kk)), 'Interpreter','tex');

    for r = 1:size(M,1)
        for c = 1:size(M,2)
            if ~isnan(M(r,c))
                text(ax, c, r, sprintf('%.3g', M(r,c)), ...
                    'HorizontalAlignment','center', 'Color','w', 'FontSize',8);
            end
        end
    end
end

fig = figure('Color','w');
t = tiledlayout(fig, 1, 5, 'TileSpacing','compact', 'Padding','compact');
title(t, sprintf("25th %%ile - Metric: %s", metric_name), 'Interpreter','none');

for kk = 1:numel(sigma)
    s = sigma(kk);
    sub = Results(Results.sigma == s, :);

    M = nan(numel(strats), numel(reward_transition));
    for a = 1:numel(strats)
        for b = 1:numel(reward_transition)
            row = sub(sub.strat==strats(a) & sub.reward_transition==reward_transition(b), :);
            if ~isempty(row)
                M(a,b) = row.perc_25th(1);
            end
        end
    end

    ax = nexttile(t, kk);
    imagesc(ax, M);
    axis(ax, 'tight');
    colormap(ax, parula);

    set(ax, ...
        'XTick', 1:numel(reward_transition), 'XTickLabel', reward_transition_names, ...
        'YTick', 1:numel(strats),           'YTickLabel', strats_names, ...
        'TickLabelInterpreter','tex');

    xlabel(ax, 'Reward Transition');
    ylabel(ax, 'Strategy');
    title(ax, sprintf("\\sigma = %s", sigma_names(kk)), 'Interpreter','tex');

    for r = 1:size(M,1)
        for c = 1:size(M,2)
            if ~isnan(M(r,c))
                text(ax, c, r, sprintf('%.3g', M(r,c)), ...
                    'HorizontalAlignment','center', 'Color','w', 'FontSize',8);
            end
        end
    end
end