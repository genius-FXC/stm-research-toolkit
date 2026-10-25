function kondoFitGui()
%KONDOFITGUI Lightweight Matlab GUI for Hurwitz-Kondo fitting.
%
% Run this file in Matlab:
%   kondoFitGui

    scriptDir = fileparts(mfilename('fullpath'));
    addpath(fullfile(fileparts(scriptDir), 'demo'), '-end');
    addpath(scriptDir, '-begin');

    state = struct();
    state.filePath = '';
    state.data = [];
    state.result = [];

    fig = figure( ...
        'Name', 'Kondo resonance fit', ...
        'NumberTitle', 'off', ...
        'MenuBar', 'none', ...
        'ToolBar', 'figure', ...
        'Color', [0.96 0.97 0.98], ...
        'Position', [120 80 1180 720]);

    left = uipanel(fig, 'Title', 'Input', 'Units', 'normalized', ...
        'Position', [0.02 0.04 0.30 0.92], 'BackgroundColor', [0.98 0.98 0.99]);
    right = uipanel(fig, 'Title', 'Preview and fit', 'Units', 'normalized', ...
        'Position', [0.34 0.04 0.64 0.92], 'BackgroundColor', [1 1 1]);
    tabGroup = uitabgroup(left, 'Units', 'normalized', 'Position', [0.02 0.02 0.96 0.96]);
    basicTab = uitab(tabGroup, 'Title', 'Basic');
    advancedTab = uitab(tabGroup, 'Title', 'Advanced');

    y = 0.93;
    label(basicTab, 'Data file', y);
    fileEdit = uicontrol(basicTab, 'Style', 'edit', 'Units', 'normalized', ...
        'Position', [0.05 y-0.06 0.66 0.045], 'HorizontalAlignment', 'left', ...
        'String', '');
    uicontrol(basicTab, 'Style', 'pushbutton', 'Units', 'normalized', ...
        'Position', [0.73 y-0.06 0.22 0.045], 'String', 'Browse', ...
        'Callback', @browseFile);

    y = y - 0.10;
    uicontrol(basicTab, 'Style', 'pushbutton', 'Units', 'normalized', ...
        'Position', [0.05 y-0.02 0.42 0.055], 'String', 'Load preview', ...
        'Callback', @loadPreview);
    uicontrol(basicTab, 'Style', 'pushbutton', 'Units', 'normalized', ...
        'Position', [0.53 y-0.02 0.42 0.055], 'String', 'Run fit', ...
        'FontWeight', 'bold', 'Callback', @runFit);
    saveButton = uicontrol(basicTab, 'Style', 'pushbutton', 'Units', 'normalized', ...
        'Position', [0.05 y-0.09 0.90 0.045], 'String', 'Save result files', ...
        'Enable', 'off', 'Callback', @saveResult);

    y = y - 0.10;
    label(basicTab, 'X column', y);
    xColEdit = editBox(basicTab, '1', [0.43 y-0.025 0.18 0.045]);
    label(basicTab, 'Y column', y-0.07);
    yColEdit = editBox(basicTab, '2', [0.43 y-0.095 0.18 0.045]);

    label(basicTab, 'X unit', y-0.14);
    unitPopup = uicontrol(basicTab, 'Style', 'popupmenu', 'Units', 'normalized', ...
        'Position', [0.43 y-0.165 0.24 0.045], 'String', {'mV','V','uV'}, 'Value', 1);

    y = y - 0.23;
    label(basicTab, 'Temperature K', y);
    tempEdit = editBox(basicTab, '4', [0.43 y-0.025 0.22 0.045]);
    label(basicTab, 'Lock-in amp', y-0.07);
    lockInEdit = editBox(basicTab, '1', [0.43 y-0.095 0.22 0.045]);
    useLiCheck = uicontrol(basicTab, 'Style', 'checkbox', 'Units', 'normalized', ...
        'Position', [0.05 y-0.16 0.70 0.05], 'String', 'Use lock-in broadening', ...
        'Value', 1, 'BackgroundColor', [0.98 0.98 0.99]);

    y = y - 0.23;
    label(basicTab, 'Fit range min', y);
    fitMinEdit = editBox(basicTab, '', [0.43 y-0.025 0.25 0.045]);
    label(basicTab, 'Fit range max', y-0.07);
    fitMaxEdit = editBox(basicTab, '', [0.43 y-0.095 0.25 0.045]);
    uicontrol(basicTab, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.05 y-0.155 0.90 0.07], 'String', ...
        'Leave range blank to use all data. Output yfit = h.fit * max(y in selected range).', ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.98 0.98 0.99], ...
        'ForegroundColor', [0.25 0.28 0.32]);

    y = y - 0.24;
    resultText = uicontrol(basicTab, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.05 0.08 0.90 y-0.02], 'String', 'No fit yet.', ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.98 0.98 0.99], ...
        'ForegroundColor', [0.10 0.12 0.16]);

    statusText = uicontrol(basicTab, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.05 0.02 0.90 0.045], 'String', 'Ready.', ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.98 0.98 0.99], ...
        'ForegroundColor', [0.12 0.25 0.45]);

    label(advancedTab, 'Model', 0.92);
    modelPopup = uicontrol(advancedTab, 'Style', 'popupmenu', 'Units', 'normalized', ...
        'Position', [0.43 0.875 0.30 0.05], 'String', {'hurwitz','frota'}, 'Value', 1);

    uicontrol(advancedTab, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.05 0.78 0.90 0.045], 'String', 'biasOffset', ...
        'HorizontalAlignment', 'left', 'FontWeight', 'bold', ...
        'BackgroundColor', [0.98 0.98 0.99], 'ForegroundColor', [0.10 0.12 0.16]);
    label(advancedTab, 'Initial', 0.74);
    biasInitEdit = editBox(advancedTab, '0', [0.43 0.715 0.22 0.045]);
    label(advancedTab, 'Lower', 0.67);
    biasLowerEdit = editBox(advancedTab, '-1', [0.43 0.645 0.22 0.045]);
    label(advancedTab, 'Upper', 0.60);
    biasUpperEdit = editBox(advancedTab, '1', [0.43 0.575 0.22 0.045]);
    biasFixCheck = uicontrol(advancedTab, 'Style', 'checkbox', 'Units', 'normalized', ...
        'Position', [0.05 0.51 0.70 0.05], 'String', 'Fix biasOffset at initial value', ...
        'Value', 0, 'BackgroundColor', [0.98 0.98 0.99]);

    uicontrol(advancedTab, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.05 0.42 0.90 0.045], 'String', 'Gamma', ...
        'HorizontalAlignment', 'left', 'FontWeight', 'bold', ...
        'BackgroundColor', [0.98 0.98 0.99], 'ForegroundColor', [0.10 0.12 0.16]);
    label(advancedTab, 'Initial', 0.38);
    gammaInitEdit = editBox(advancedTab, '1', [0.43 0.355 0.22 0.045]);
    label(advancedTab, 'Lower', 0.31);
    gammaLowerEdit = editBox(advancedTab, '0', [0.43 0.285 0.22 0.045]);
    label(advancedTab, 'Upper', 0.24);
    gammaUpperEdit = editBox(advancedTab, '25.42', [0.43 0.215 0.22 0.045]);

    uicontrol(advancedTab, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.05 0.08 0.90 0.10], 'String', ...
        'Gamma is entered in the same unit as x. 25.42 matches the source-program upper bound; larger values may change the local optimum.', ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.98 0.98 0.99], ...
        'ForegroundColor', [0.25 0.28 0.32]);

    fitSummaryEdit = uicontrol(right, 'Style', 'edit', 'Units', 'normalized', ...
        'Position', [0.05 0.91 0.90 0.045], 'String', 'Fit results will appear here.', ...
        'HorizontalAlignment', 'left', 'Enable', 'inactive', ...
        'BackgroundColor', [0.98 0.98 0.99]);

    ax = axes(right, 'Units', 'normalized', 'Position', [0.08 0.38 0.88 0.50]);
    grid(ax, 'on');
    xlabel(ax, 'x');
    ylabel(ax, 'y');
    title(ax, 'Data preview');

    dataTable = uitable(right, 'Units', 'normalized', 'Position', [0.05 0.05 0.90 0.27]);

    function browseFile(~, ~)
        [fileName, folder] = uigetfile( ...
            {'*.txt;*.dat;*.csv;*.xlsx;*.xls', 'Data files (*.txt, *.dat, *.csv, *.xlsx, *.xls)'; '*.*', 'All files'}, ...
            'Select Kondo data file', scriptDir);
        if isequal(fileName, 0)
            return;
        end
        state.filePath = fullfile(folder, fileName);
        set(fileEdit, 'String', state.filePath);
        loadPreview();
    end

    function loadPreview(~, ~)
        filePath = strtrim(get(fileEdit, 'String'));
        if isempty(filePath)
            setStatus('Choose a data file first.');
            return;
        end
        try
            state.filePath = filePath;
            state.data = readmatrix(filePath);
            if isempty(state.data)
                error('The selected file has no numeric data.');
            end
            previewRows = min(20, size(state.data, 1));
            set(dataTable, 'Data', state.data(1:previewRows, :));
            plotSelectedColumns(false);
            setStatus(sprintf('Loaded %d rows and %d columns.', size(state.data, 1), size(state.data, 2)));
        catch err
            setStatus(['Load failed: ' err.message]);
        end
    end

    function runFit(~, ~)
        filePath = strtrim(get(fileEdit, 'String'));
        if isempty(filePath)
            setStatus('Choose a data file first.');
            return;
        end

        try
            setStatus('Fitting. Matlab may take a moment...');
            drawnow;

            opts = struct();
            opts.xColumn = readPositiveInteger(xColEdit, 'X column');
            opts.yColumn = readPositiveInteger(yColEdit, 'Y column');
            opts.temperature = readNumber(tempEdit, 'Temperature');
            opts.lockInAmplitude = readNumber(lockInEdit, 'Lock-in amplitude');
            opts.useLockIn = logical(get(useLiCheck, 'Value'));
            units = get(unitPopup, 'String');
            opts.units = units{get(unitPopup, 'Value')};
            models = get(modelPopup, 'String');
            opts.model = models{get(modelPopup, 'Value')};
            opts.fitMin = readOptionalNumber(fitMinEdit);
            opts.fitMax = readOptionalNumber(fitMaxEdit);
            opts.biasOffsetInitial = readNumber(biasInitEdit, 'biasOffset initial');
            opts.biasOffsetLower = readNumber(biasLowerEdit, 'biasOffset lower');
            opts.biasOffsetUpper = readNumber(biasUpperEdit, 'biasOffset upper');
            opts.fixBiasOffset = logical(get(biasFixCheck, 'Value'));
            opts.gammaInitial = readNumber(gammaInitEdit, 'Gamma initial');
            opts.gammaLower = readNumber(gammaLowerEdit, 'Gamma lower');
            opts.gammaUpper = readNumber(gammaUpperEdit, 'Gamma upper');
            if ~isempty(opts.fitMin) && ~isempty(opts.fitMax) && opts.fitMin > opts.fitMax
                tmp = opts.fitMin;
                opts.fitMin = opts.fitMax;
                opts.fitMax = tmp;
            end
            opts.outputDir = fullfile(fileparts(scriptDir), 'output');
            opts.closeClassFigure = true;
            opts.saveOutput = false;

            state.result = runKondoFit(filePath, opts);
            plotResult();
            showResult();
            set(saveButton, 'Enable', 'on');
            setStatus('Fit complete.');
        catch err
            setStatus(['Fit failed: ' err.message]);
        end
    end

    function saveResult(~, ~)
        if isempty(state.result)
            setStatus('Run a fit before saving.');
            return;
        end
        try
            outputDir = fullfile(fileparts(scriptDir), 'output');
            state.result.outputFiles = saveKondoFitResult(state.result, state.result.filePath, outputDir);
            showResult();
            setStatus(['Saved result files to ' outputDir]);
        catch err
            setStatus(['Save failed: ' err.message]);
        end
    end

    function plotSelectedColumns(keepTitle)
        xCol = readPositiveInteger(xColEdit, 'X column');
        yCol = readPositiveInteger(yColEdit, 'Y column');
        if xCol > size(state.data, 2) || yCol > size(state.data, 2)
            error('Selected x/y columns are outside the data matrix.');
        end
        x = state.data(:, xCol);
        yv = state.data(:, yCol);
        valid = isfinite(x) & isfinite(yv);
        plot(ax, x(valid), yv(valid), 'Color', [0.10 0.35 0.65], 'LineWidth', 1.2);
        grid(ax, 'on');
        xlabel(ax, 'x');
        ylabel(ax, 'y');
        if ~keepTitle
            title(ax, 'Data preview');
        end
    end

    function plotResult()
        r = state.result;
        plot(ax, r.x, r.y, '-', 'Color', [0.10 0.35 0.65], 'LineWidth', 1.1);
        hold(ax, 'on');
        plot(ax, r.x, r.yfit, '--', 'Color', [0.80 0.20 0.12], 'LineWidth', 1.7);
        hold(ax, 'off');
        grid(ax, 'on');
        xlabel(ax, ['x (' r.units ')']);
        ylabel(ax, 'y');
        title(ax, sprintf('Gamma = %.6g', r.Gamma));
        legend(ax, {'data used for fit', 'yfit'}, 'Location', 'best');
        outData = [r.x(:), r.y(:), r.yfit(:), r.fitNormalized(:)];
        set(dataTable, 'Data', outData);
        set(dataTable, 'ColumnName', {'x','y','yfit','fitNormalized'});
        set(fitSummaryEdit, 'String', sprintf('Gamma = %.15g    biasOffset = %.15g    RMSE = %.6g', ...
            r.Gamma, r.biasOffset, r.rmseNormalized));
    end

    function showResult()
        r = state.result;
        if isfield(r, 'outputFiles') && ~isempty(r.outputFiles.xyfit)
            savedText = sprintf('Saved:\n%s\n%s\n%s', ...
                r.outputFiles.xyfit, r.outputFiles.full, r.outputFiles.params);
        else
            savedText = 'Not saved yet. Click "Save result files" to write output.';
        end
        msg = sprintf([ ...
            'Gamma: %.15g\n' ...
            'biasOffset: %.15g\n' ...
            'phi: %.15g\n' ...
            'model: %s\n' ...
            'normalized RMSE: %.6g\n' ...
            'y max used: %.15g\n\n' ...
            '%s'], ...
            r.Gamma, r.biasOffset, r.phi, r.model, r.rmseNormalized, r.yMax, savedText);
        set(resultText, 'String', msg);
    end

    function setStatus(message)
        set(statusText, 'String', message);
        drawnow;
    end
end

function label(parent, text, y)
    uicontrol(parent, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.05 y-0.02 0.36 0.04], 'String', text, ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.98 0.98 0.99], ...
        'ForegroundColor', [0.10 0.12 0.16]);
end

function h = editBox(parent, value, pos)
    h = uicontrol(parent, 'Style', 'edit', 'Units', 'normalized', ...
        'Position', pos, 'String', value, 'HorizontalAlignment', 'left');
end

function value = readPositiveInteger(control, name)
    value = str2double(strtrim(get(control, 'String')));
    if ~isfinite(value) || value < 1 || round(value) ~= value
        error('%s must be a positive integer.', name);
    end
end

function value = readNumber(control, name)
    value = str2double(strtrim(get(control, 'String')));
    if ~isfinite(value)
        error('%s must be a finite number.', name);
    end
end

function value = readOptionalNumber(control)
    text = strtrim(get(control, 'String'));
    if isempty(text)
        value = [];
        return;
    end
    value = str2double(text);
    if ~isfinite(value)
        error('Fit range values must be numbers or blank.');
    end
end
