clc; clear; %close all;

% ======================
% USER INPUTS
% ======================
data_dir = "data_CVaR_ablation";

k_names = ["7","15"];     % plot both on same axes
sigma_name = "5_0";

metrics_to_plot = ["Mean","PCT"];   % pick any subset: ["Mean"], ["PCT","CE"], etc.

% Numeric alphas (x-axis)
CVaR_alphas_test = logspace(log10(1e-3), log10(0.99), 72);

% Filename alpha names (MUST match your Julia names)
CVaR_alphas_test_names = ["0_1", "0_11020265178730582", "0_12144624460954163", "0_13383698205581282", "0_14749190329760636", "0_1625399886055308", "0_17912337765807965", "0_19739871215019425", "0_21753861538350466", "0_2397333228140099", "0_2641924789588611", "0_2911471176352849", "0_32085184423639024", "0_353587240656978", "0_3896625155855525", "0_4294184251964021", "0_4732304918297234", "0_5215125510624646", "0_5747206606744634", "0_6333574084327817", "0_6979766593842825", "0_7691887874979305", "0_8476664410733432", "0_9341508963739041", "1_029459059498929", "1_134491182632478", "1_2502393675541568", "1_3777969367335217", "1_5183687605246083", "1_6732826380081647", "1_8440018389815824", "2_0321389255643876", "2_2394709839740194", "2_467956410346639", "2_719753409156797", "2_9972403789664437", "3_3030383780609136", "3_640035882175542", "4_011416068166895", "4_420686881341996", "4_871714170452428", "5_368758203336521", "5_916513908125362", "6_520155220118908", "7_185383953219477", "7_9184836575474105", "8_726378971961685", "9_616701032111612", "10_597859551844198", "11_679122258826586", "12_870702434708386", "14_183855386701966", "15_63098476182218", "17_225759707997724", "18_983243988722755", "20_92003827082679", "23_054436929370354", "25_406600850798043", "27_998747866595636", "30_855362616230092", "34_00342782167457", "37_472679158067876", "41_29588612793985", "45_50916159205586", "50_152302880615615", "55_26916770683976", "60_908088433710674", "67_12232860690644", "73_97058606620024", "81_51754738756397", "89_8344988930691", "99"];
% e.g. CVaR_alphas_test_names = ["0_1","0_12",...];

% Metrics
beta = 0.5;   % entropic score parameter
pct  = 1;     % Xth percentile (e.g. 1, 5, 10)

metricCE  = @(v) -(1/beta) * log(mean(exp(-beta*v), 'omitnan') + eps);
metricPCT = @(v) prctile(v, pct);

% ======================
% LOAD METRICS (vectors)
% ======================
na = numel(CVaR_alphas_test);
nk = numel(k_names);

Mean = nan(na,nk,2);  % (:,k,1)=non, (:,k,2)=expl
PCTv = nan(na,nk,2);
CE   = nan(na,nk,2);

for ik = 1:nk
    k_name = k_names(ik);

    for ia = 1:na
        aName = CVaR_alphas_test_names(ia);

        f_non = fullfile(data_dir, "batch_CVaR_alpha" + aName + "_k" + k_name + "_sigma" + sigma_name + ".csv");
        f_exp = fullfile(data_dir, "batch_CVaR_alpha" + aName + "_expl_k" + k_name + "_sigma" + sigma_name + ".csv");

        [Mean(ia,ik,1), PCTv(ia,ik,1), CE(ia,ik,1)] = read_metrics(f_non, metricPCT, metricCE);
        [Mean(ia,ik,2), PCTv(ia,ik,2), CE(ia,ik,2)] = read_metrics(f_exp, metricPCT, metricCE);
    end
end

% ======================
% PLOTS (3 figures)
% ======================
% ======================
% SUBPLOTS + COMMON LEGEND
% ======================
fig = figure('Color','w');
t = tiledlayout(fig, numel(metrics_to_plot), 1, 'TileSpacing','compact', 'Padding','compact');

lgd_handles = gobjects(0);
lgd_labels  = strings(0);

for mi = 1:numel(metrics_to_plot)
    ax = nexttile(t, mi);
    hold(ax,'on'); grid(ax,'on');
    set(ax,'XScale','log');
    xlabel(ax, '\alpha (CVaR)');

    which = metrics_to_plot(mi);

    switch which
        case "Mean"
            Y = Mean;
            ylab = "Mean landing value";
            y_lims = [];
        case "PCT"
            Y = PCTv;
            ylab = sprintf("P%02d landing value", pct);
            y_lims = [];
        case "CE"
            Y = CE;
            ylab = sprintf("Entropic score (CE), \\beta=%.3g", beta);
            y_lims = [8 14];   % optional; set [] for auto
        otherwise
            error("Unknown metric: %s", which);
    end

    ylabel(ax, ylab);

    [h, labels] = plot_k_two_lines_on_ax(ax, CVaR_alphas_test, Y, k_names);

    % collect legend entries once (from first subplot only)
    if mi == 1
        lgd_handles = h;
        lgd_labels  = labels;
    end

    if ~isempty(y_lims)
        ylim(ax, y_lims);
    end
end

% Create one legend tied to the last active axes (or pick the first)
lgd = legend(ax, lgd_handles, lgd_labels, 'Orientation','horizontal');
lgd.Location = 'southoutside';

% ======================
% Local functions
% ======================
function [m, p, ce] = read_metrics(filename, metricPCT, metricCE)
    if ~isfile(filename)
        m = NaN; p = NaN; ce = NaN;
        return;
    end
    T = readtable(filename);
    v = T{:,8};
    v = v(:);
    v = v(~isnan(v));
    if isempty(v)
        m = NaN; p = NaN; ce = NaN;
    else
        m  = mean(v,'omitnan');
        p  = metricPCT(v);
        ce = metricCE(v);
    end
end

function [h, labels] = plot_k_two_lines_on_ax(ax, alphas, Y, k_names)
% Y is na x nk x 2 (non/expl). Returns line handles + labels for legend.
    colors = lines(numel(k_names));

    h = gobjects(0);
    labels = strings(0);

    for ik = 1:numel(k_names)
        y_non = Y(:,ik,1);
        y_exp = Y(:,ik,2);

        h(end+1) = plot(ax, alphas, y_non, '-',  'Color', colors(ik,:), 'LineWidth', 2); %#ok<AGROW>
        labels(end+1) = "k=" + k_names(ik) + " no-expl"; %#ok<AGROW>

        h(end+1) = plot(ax, alphas, y_exp, '--', 'Color', colors(ik,:), 'LineWidth', 2); %#ok<AGROW>
        labels(end+1) = "k=" + k_names(ik) + " expl"; %#ok<AGROW>
    end
end
