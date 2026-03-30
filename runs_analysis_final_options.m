clc; clear; close all;

% =============================
%  USER SETTINGS (parallel)
% =============================

plot_type = "heatmap_single_regime"; %"heatmap", "heatmap_single_regime", "scatter_per_regime", "scatter_all_regimes", "first_two", "first_three", "all"

% Pick one regime to show (must match your Results.reward_transition / Results.sigma values)
pick_k    = "k15";
pick_sigma = "5_0";

%{
strats = [..."max_perc", "min_perc", "mean_perc", "cellwise_perc",...
          ..."CVaR_alpha0_1", "CVaR_alpha1", "CVaR_alpha10", "CVaR_alpha50",...
          "EVaR_alpha0_5", "EVaR_alpha5", "EVaR_alpha10", "EVaR_alpha25", "EVaR_alpha50", "EVaR_alpha90",...
          ..."max_perc_expl", "min_perc_expl", "mean_perc_expl", "cellwise_perc_expl",...
          ..."CVaR_alpha0_1_expl", "CVaR_alpha1_expl", "CVaR_alpha10_expl", "CVaR_alpha50_expl",...
          "EVaR_alpha0_5_expl", "EVaR_alpha5_expl", "EVaR_alpha10_expl", "EVaR_alpha25_expl", "EVaR_alpha50_expl", "EVaR_alpha90_expl",...
          ];

strats_names = [..."Max \sigma", "Min \sigma", "Mean \sigma", "C-wise \sigma",...
                ..."CVaR \alpha 0.001", "CVaR \alpha 0.01", "CVaR \alpha 0.1", "CVaR \alpha 0.5",...
                "EVaR \alpha 0.005", "EVaR \alpha 0.05", "EVaR \alpha 0.1", "EVaR \alpha 0.25", "EVaR \alpha 0.5", "EVaR \alpha 0.9",...
                ..."Max \sigma expl", "Min \sigma expl", "Mean \sigma expl", "C-wise \sigma expl",...
                ..."CVaR \alpha 0.001 expl", "CVaR \alpha 0.01 expl", "CVaR \alpha 0.1 expl", "CVaR \alpha 0.5 expl",...
                "EVaR \alpha 0.005 expl", "EVaR \alpha 0.05 expl", "EVaR \alpha 0.1 expl", "EVaR \alpha 0.25 expl", "EVaR \alpha 0.5 expl", "EVaR \alpha 0.9 expl";...
                ];

%}

%%{
strats = ["max_perc", "cellwise_perc",...
          "const_p50",...
          "CVaR_alpha80",...
          "max_perc_expl", "cellwise_perc_expl",...
          "const_p50_expl",...
          "CVaR_alpha25_expl",...
          ];

strats_names = ["Max \sigma", "C-wise \sigma",...
                "Mean Targeting",...
                "CVaR \alpha 0.80",...
                "Max \sigma expl", "C-wise \sigma expl",...
                "Mean Targeting expl",...
                "CVaR \alpha 0.25 expl",...
                ];
%}

reward_transition = ["k0", "k4", "k7", "k15"];
reward_transition_names = ["k=0", "k=4", "k=7", "k=15"];

sigma = ["0_5", "2_0", "5_0"];
sigma_names = ["0.5", "2.0", "5.0"];

data_dir = "data_final";  % change if needed

% Metrics
beta = 0.5;
metricCE  = @(v) -(1/beta) * log(mean(exp(-beta * v), 'omitnan') + eps);
metricP01 = @(v) prctile(v, 1);

% =============================
%  LOAD + COMPUTE RESULTS TABLE
% =============================
n = numel(strats) * numel(reward_transition) * numel(sigma);
Results = table( ...
    strings(n,1), strings(n,1), strings(n,1), ...
    nan(n,1), nan(n,1), nan(n,1), ...
    'VariableNames', {'strat','reward_transition','sigma','Mean','CE','P01'} );

idx = 0;
for i = 1:numel(strats)
    for j = 1:numel(reward_transition)
        for k = 1:numel(sigma)
            idx = idx + 1;

            st = strats(i);
            kk = reward_transition(j);
            ss = sigma(k);

            filename = fullfile(data_dir, sprintf("batch_%s_%s_sigma%s.csv", st, kk, ss));
            Results.strat(idx) = st;
            Results.reward_transition(idx) = kk;
            Results.sigma(idx) = ss;

            if ~isfile(filename)
                Results.Mean(idx) = NaN;
                Results.CE(idx)   = NaN;
                Results.P01(idx)  = NaN;
                continue;
            end

            T = readtable(filename);
            v = T{:,8};
            v = v(:);
            v = v(~isnan(v));

            if isempty(v)
                Results.Mean(idx) = NaN;
                Results.CE(idx)   = NaN;
                Results.P01(idx)  = NaN;
            else
                Results.Mean(idx) = mean(v, 'omitnan');
                Results.CE(idx)   = metricCE(v);
                Results.P01(idx)  = metricP01(v);
            end
        end
    end
end

% =============================
%  HEATMAPS: 9 regimes
%  (one subplot per regime)
% =============================
if (plot_type == "heatmap") || (plot_type == "first_two") || (plot_type == "first_three") || (plot_type == "all")
    make_heatmap_grid(Results, strats, strats_names, reward_transition, reward_transition_names, sigma, sigma_names, "CE");
    make_heatmap_grid(Results, strats, strats_names, reward_transition, reward_transition_names, sigma, sigma_names, "P01");
    make_heatmap_grid(Results, strats, strats_names, reward_transition, reward_transition_names, sigma, sigma_names, "Mean");
end

if (plot_type == "heatmap_single_regime") || (plot_type == "first_two") || (plot_type == "first_three") || (plot_type == "all")
    make_heatmap_three_metrics_single_regime(Results, strats, strats_names, ...
        pick_k, pick_sigma, "Mean", "P01", "CE", beta);
end

% =============================
%  SCATTER: mean vs P01 / CE
%  Option A: one figure per regime (9 figures)
% =============================
if (plot_type == "scatter_per_regime") || (plot_type == "first_three") || (plot_type == "all")
    plot_scatter_per_regime(Results, strats, strats_names, reward_transition, reward_transition_names, sigma, sigma_names, "P01");
    plot_scatter_per_regime(Results, strats, strats_names, reward_transition, reward_transition_names, sigma, sigma_names, "CE");
end

% =============================
%  SCATTER: all regimes combined
%  (color by regime, marker by strategy)
% =============================
if (plot_type == "scatter_all_regimes") || (plot_type == "all")
    plot_scatter_all_regimes(Results, strats, strats_names, reward_transition, reward_transition_names, sigma, sigma_names, "P01");
    plot_scatter_all_regimes(Results, strats, strats_names, reward_transition, reward_transition_names, sigma, sigma_names, "CE");
end

% ======================================================================
%  Helper functions (MATLAB allows local functions at end of script)
% ======================================================================

function make_heatmap_grid(Results, strats, strats_names, reward_transition, reward_transition_names, sigma, sigma_names, field)
    fig = figure('Color','w');
    t = tiledlayout(fig, numel(sigma), numel(reward_transition), 'TileSpacing','compact', 'Padding','compact');
    title(t, sprintf("%s by Strategy (rows) for 9 regimes", field), 'Interpreter','none');

    % Determine global color range for comparability across tiles
    vals = Results.(field);
    clim = [min(vals, [], 'omitnan'), max(vals, [], 'omitnan')];

    for si = 1:numel(sigma)
        for ki = 1:numel(reward_transition)
            ss = sigma(si);
            kk = reward_transition(ki);

            sub = Results(Results.sigma == ss & Results.reward_transition == kk, :);

            M = nan(numel(strats), 1);
            for a = 1:numel(strats)
                row = sub(sub.strat == strats(a), :);
                if ~isempty(row)
                    M(a,1) = row.(field)(1);
                end
            end

            ax = nexttile(t, (si-1)*numel(reward_transition) + ki);
            imagesc(ax, M);
            colormap(ax, parula);
            axis(ax, 'tight');
            set(ax, 'CLim', clim);

            set(ax, 'YTick', 1:numel(strats), 'YTickLabel', strats_names, 'TickLabelInterpreter','tex');
            set(ax, 'XTick', 1, 'XTickLabel', {''});

            title(ax, sprintf("\\sigma=%s, %s", sigma_names(si), reward_transition_names(ki)), 'Interpreter','tex');

            for r = 1:size(M,1)
                if ~isnan(M(r,1))
                    text(ax, 1, r, sprintf('%.3g', M(r,1)), ...
                        'HorizontalAlignment','center', 'Color','w', 'FontSize',8);
                end
            end
        end
    end
    cb = colorbar; %#ok<NASGU>
end


function plot_scatter_per_regime(Results, strats, strats_names, reward_transition, reward_transition_names, sigma, sigma_names, yfield)

    fig = figure('Color','w');
    t = tiledlayout(fig, numel(sigma), numel(reward_transition), ...
        'TileSpacing','compact', 'Padding','compact');
    title(t, sprintf("Mean vs %s (all 9 regimes)", yfield), 'Interpreter','none');

    for si = 1:numel(sigma)
        for ki = 1:numel(reward_transition)
            ss = sigma(si);
            kk = reward_transition(ki);

            sub = Results(Results.sigma == ss & Results.reward_transition == kk, :);

            ax = nexttile(t, (si-1)*numel(reward_transition) + ki);
            hold(ax,'on'); grid(ax,'on');
            title(ax, sprintf("\\sigma=%s, %s", sigma_names(si), reward_transition_names(ki)), 'Interpreter','tex');
            xlabel(ax, "Mean landing value");
            ylabel(ax, yfield);

            for a = 1:numel(strats)
                row = sub(sub.strat == strats(a), :);
                if isempty(row); continue; end

                x = row.Mean(1);
                y = row.(yfield)(1);
                if isnan(x) || isnan(y); continue; end

                scatter(ax, x, y, 60, 'filled');
                text(ax, x, y, sprintf("  %s", strats_names(a)), 'Interpreter','tex');
            end
        end
    end
end

function plot_scatter_all_regimes(Results, strats, strats_names, reward_transition, reward_transition_names, sigma, sigma_names, yfield)
    fig = figure('Color','w'); hold on; grid on;
    title(sprintf("Mean vs %s (all 9 regimes)", yfield), 'Interpreter','none');
    xlabel("Mean landing value");
    ylabel(yfield);

    % Color by regime (sigma,k); marker by strategy
    colors = lines(numel(sigma)*numel(reward_transition));
    markers = {'o','s','^','d','v','>','<','p','h','x','+'};

    reg_idx = 0;
    for si = 1:numel(sigma)
        for ki = 1:numel(reward_transition)
            reg_idx = reg_idx + 1;
            ss = sigma(si);
            kk = reward_transition(ki);
            reg = Results(Results.sigma == ss & Results.reward_transition == kk, :);

            for a = 1:numel(strats)
                row = reg(reg.strat == strats(a), :);
                if isempty(row); continue; end
                x = row.Mean(1);
                y = row.(yfield)(1);
                if isnan(x) || isnan(y); continue; end

                mk = markers{1 + mod(a-1, numel(markers))};
                scatter(x, y, 70, 'Marker', mk, 'MarkerEdgeColor', colors(reg_idx,:), ...
                    'DisplayName', sprintf("%s | \\sigma=%s %s", strats_names(a), sigma_names(si), reward_transition_names(ki)));
            end
        end
    end
    legend('Location','bestoutside');
end

function make_heatmap_three_metrics_single_regime(Results, strats, strats_names, ...
        pick_k, pick_sigma, field1, field2, field3, beta)

    fields = [string(field1), string(field2), string(field3)];
    titles = strings(1,3);
    for i = 1:3
        switch fields(i)
            case "Mean"
                titles(i) = "Mean";
            case "P01"
                titles(i) = "1st %ile";
            case "CE"
                titles(i) = sprintf("CE, \\beta=%.3g", beta);
            otherwise
                titles(i) = fields(i);
        end
    end

    fig = figure('Color','w');
    t = tiledlayout(fig, 1, 3, 'TileSpacing','compact', 'Padding','compact');
    title(t, sprintf("Regime: %s, \\sigma=%s", pick_k, pick_sigma), 'Interpreter','tex');

    sub = Results(Results.reward_transition == pick_k & Results.sigma == pick_sigma, :);

    for fi = 1:3
        field = fields(fi);

        % Build vector over strategies
        M = nan(numel(strats), 1);
        for a = 1:numel(strats)
            row = sub(sub.strat == strats(a), :);
            if ~isempty(row)
                M(a,1) = row.(field)(1);
            end
        end

        ax = nexttile(t, fi);
        imagesc(ax, M);
        colormap(ax, parula);
        axis(ax, 'tight');

        % Separate color scaling PER COLUMN (metric)
        clim = [min(M, [], 'omitnan'), max(M, [], 'omitnan')];
        if any(isnan(clim)) || clim(1) == clim(2)
            clim = clim + [-1, 1]; % avoid degenerate CLim
        end
        set(ax, 'CLim', clim);

        set(ax, 'YTick', 1:numel(strats), 'YTickLabel', strats_names, 'TickLabelInterpreter','tex');
        set(ax, 'XTick', 1, 'XTickLabel', {''});

        title(ax, titles(fi), 'Interpreter','tex');

        for r = 1:numel(M)
            if ~isnan(M(r))
                text(ax, 1, r, sprintf('%.3g', M(r)), ...
                    'HorizontalAlignment','center', 'Color','w', 'FontSize',8);
            end
        end

        colorbar(ax); % one colorbar per metric/column
    end
end