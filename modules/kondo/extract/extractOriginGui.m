function extractOriginGui()
%EXTRACTORIGINGUI Extract curve data from an Origin project into txt files.
%
% Workflow:
%   1. Save the embedded Origin object from PowerPoint as .opj/.opju.
%   2. Run this GUI.
%   3. Load the Origin file, select curves, and export each curve to txt.

    scriptDir = fileparts(mfilename('fullpath'));
    inputDir = fullfile(scriptDir, 'input');
    outputDirDefault = fullfile(scriptDir, 'output');
    if ~isfolder(inputDir), mkdir(inputDir); end
    if ~isfolder(outputDirDefault), mkdir(outputDirDefault); end

    state = struct();
    state.originFile = '';
    state.outputDir = outputDirDefault;
    state.curves = struct([]);
    state.app = [];

    fig = figure( ...
        'Name', 'Origin curve extractor', ...
        'NumberTitle', 'off', ...
        'MenuBar', 'none', ...
        'ToolBar', 'figure', ...
        'Color', [0.96 0.97 0.98], ...
        'Position', [120 80 1220 740], ...
        'CloseRequestFcn', @closeGui);

    controls = uipanel(fig, 'Title', 'Extract', 'Units', 'normalized', ...
        'Position', [0.02 0.04 0.32 0.92], 'BackgroundColor', [0.98 0.98 0.99]);
    preview = uipanel(fig, 'Title', 'Preview', 'Units', 'normalized', ...
        'Position', [0.36 0.04 0.62 0.92], 'BackgroundColor', [1 1 1]);

    y = 0.93;
    label(controls, 'Origin file', y);
    fileEdit = uicontrol(controls, 'Style', 'edit', 'Units', 'normalized', ...
        'Position', [0.05 y-0.06 0.66 0.045], 'String', '', ...
        'HorizontalAlignment', 'left');
    uicontrol(controls, 'Style', 'pushbutton', 'Units', 'normalized', ...
        'Position', [0.73 y-0.06 0.22 0.045], 'String', 'Browse', ...
        'Callback', @browseOrigin);

    y = y - 0.10;
    label(controls, 'Column mode', y);
    modePopup = uicontrol(controls, 'Style', 'popupmenu', 'Units', 'normalized', ...
        'Position', [0.36 y-0.025 0.59 0.05], ...
        'String', {'Adjacent X/Y pairs', 'Column 1 as X for all Y'}, 'Value', 1);

    y = y - 0.09;
    multiplyXCheck = uicontrol(controls, 'Style', 'checkbox', 'Units', 'normalized', ...
        'Position', [0.05 y-0.02 0.42 0.05], 'String', 'Multiply x', ...
        'Value', 1, 'BackgroundColor', [0.98 0.98 0.99], ...
        'Callback', @scaleControlChanged);
    multiplyXEdit = uicontrol(controls, 'Style', 'edit', 'Units', 'normalized', ...
        'Position', [0.53 y-0.015 0.42 0.045], 'String', '1000', ...
        'HorizontalAlignment', 'left', 'Callback', @scaleControlChanged);

    y = y - 0.06;
    multiplyYCheck = uicontrol(controls, 'Style', 'checkbox', 'Units', 'normalized', ...
        'Position', [0.05 y-0.02 0.42 0.05], 'String', 'Multiply y', ...
        'Value', 0, 'BackgroundColor', [0.98 0.98 0.99], ...
        'Callback', @scaleControlChanged);
    multiplyYEdit = uicontrol(controls, 'Style', 'edit', 'Units', 'normalized', ...
        'Position', [0.53 y-0.015 0.42 0.045], 'String', '1', ...
        'HorizontalAlignment', 'left', 'Enable', 'off', ...
        'Callback', @scaleControlChanged);

    y = y - 0.07;
    label(controls, 'Output folder', y);
    outputEdit = uicontrol(controls, 'Style', 'edit', 'Units', 'normalized', ...
        'Position', [0.05 y-0.06 0.66 0.045], 'String', state.outputDir, ...
        'HorizontalAlignment', 'left');
    uicontrol(controls, 'Style', 'pushbutton', 'Units', 'normalized', ...
        'Position', [0.73 y-0.06 0.22 0.045], 'String', 'Choose', ...
        'Callback', @chooseOutput);

    y = y - 0.10;
    uicontrol(controls, 'Style', 'pushbutton', 'Units', 'normalized', ...
        'Position', [0.05 y-0.02 0.42 0.055], 'String', 'Load curves', ...
        'Callback', @loadCurves);
    uicontrol(controls, 'Style', 'pushbutton', 'Units', 'normalized', ...
        'Position', [0.53 y-0.02 0.42 0.055], 'String', 'Export selected', ...
        'FontWeight', 'bold', 'Callback', @exportSelected);

    y = y - 0.09;
    uicontrol(controls, 'Style', 'pushbutton', 'Units', 'normalized', ...
        'Position', [0.05 y-0.02 0.28 0.045], 'String', 'All', ...
        'Callback', @(~,~) setAllIncluded(true));
    uicontrol(controls, 'Style', 'pushbutton', 'Units', 'normalized', ...
        'Position', [0.36 y-0.02 0.28 0.045], 'String', 'None', ...
        'Callback', @(~,~) setAllIncluded(false));
    uicontrol(controls, 'Style', 'pushbutton', 'Units', 'normalized', ...
        'Position', [0.67 y-0.02 0.28 0.045], 'String', 'Preview', ...
        'Callback', @plotPreview);

    curveTable = uitable(controls, 'Units', 'normalized', ...
        'Position', [0.05 0.13 0.90 0.25], ...
        'ColumnName', {'Use','Export name','Source','Pts','Y range','Y std'}, ...
        'ColumnEditable', [true true false false false false], ...
        'CellEditCallback', @tableEdited);

    statusText = uicontrol(controls, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.05 0.02 0.90 0.08], 'String', 'Ready.', ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.98 0.98 0.99], ...
        'ForegroundColor', [0.12 0.25 0.45]);

    ax = axes(preview, 'Units', 'normalized', 'Position', [0.08 0.36 0.88 0.58]);
    grid(ax, 'on');
    xlabel(ax, 'x');
    ylabel(ax, 'y');
    title(ax, 'Curves from one Origin file use shared axes here.');

    sampleTable = uitable(preview, 'Units', 'normalized', ...
        'Position', [0.05 0.05 0.90 0.24], ...
        'ColumnName', {'curve','x','y'});

    function browseOrigin(~, ~)
        [fileName, folder] = uigetfile( ...
            {'*.opju;*.opj', 'Origin projects (*.opju, *.opj)'; '*.*', 'All files'}, ...
            'Select Origin project', inputDir);
        if isequal(fileName, 0), return; end
        state.originFile = fullfile(folder, fileName);
        set(fileEdit, 'String', state.originFile);
        loadCurves();
    end

    function chooseOutput(~, ~)
        selected = uigetdir(state.outputDir, 'Select output folder');
        if isequal(selected, 0), return; end
        state.outputDir = selected;
        set(outputEdit, 'String', state.outputDir);
    end

    function loadCurves(~, ~)
        filePath = strtrim(get(fileEdit, 'String'));
        if isempty(filePath)
            setStatus('Choose an Origin file first.');
            return;
        end

        try
            setStatus('Opening Origin and reading worksheets...');
            drawnow;
            state.originFile = filePath;
            state.outputDir = strtrim(get(outputEdit, 'String'));

            closeOriginApp();
            state.app = actxserver('Origin.ApplicationSI');
            invoke(state.app, 'Load', filePath);

            modeValue = get(modePopup, 'Value');
            state.curves = readOriginCurves(state.app, modeValue);
            if isempty(state.curves)
                setStatus('No worksheet curves found. Trying graph layer plots...');
                drawnow;
                state.curves = readGraphCurves(state.app);
            end

            if isempty(state.curves)
                setStatus('No numeric curves found in worksheets or graph layer plots.');
            else
                refreshCurveTable();
                plotPreview();
                setStatus(sprintf('Loaded %d candidate curves.', numel(state.curves)));
            end
        catch err
            setStatus(['Load failed: ' err.message]);
        end
    end

    function curves = readOriginCurves(app, modeValue)
        booksText = char(invoke(app, 'ProjectSearch', 'book'));
        bookTokens = regexp(booksText, '\[([^\]]+)\]', 'tokens');
        bookNames = unique(cellfun(@(t) t{1}, bookTokens, 'UniformOutput', false), 'stable');
        curves = struct('name', {}, 'exportName', {}, 'book', {}, 'sheet', {}, 'xCol', {}, 'yCol', {}, ...
            'x', {}, 'y', {}, 'include', {}, 'yRange', {}, 'yStd', {});

        for b = 1:numel(bookNames)
            book = bookNames{b};
            emptySheetStreak = 0;
            for sheetIndex = 1:50
                ref = sprintf('[%s]%d', book, sheetIndex);
                try
                    wks = invoke(app, 'FindWorksheet', ref);
                    sheetName = char(get(wks, 'Name'));
                    nCols = double(get(wks, 'Cols'));
                    nRows = double(get(wks, 'Rows'));
                catch
                    emptySheetStreak = emptySheetStreak + 1;
                    if emptySheetStreak >= 3, break; end
                    continue;
                end

                if isempty(sheetName) || ~isfinite(nCols) || nCols < 2 || ~isfinite(nRows) || nRows < 2
                    emptySheetStreak = emptySheetStreak + 1;
                    if emptySheetStreak >= 3, break; end
                    continue;
                end
                emptySheetStreak = 0;

                invoke(wks, 'Activate');
                data = readActiveWorksheetMatrix(app, nRows, nCols);
                if isempty(data), continue; end

                if modeValue == 1
                    pairs = [(1:2:nCols-1).', (2:2:nCols).'];
                else
                    pairs = [ones(max(nCols-1, 0), 1), (2:nCols).'];
                end

                for p = 1:size(pairs, 1)
                    xCol = pairs(p, 1);
                    yCol = pairs(p, 2);
                    x = data(:, xCol);
                    y = data(:, yCol);
                    valid = isfinite(x) & isfinite(y);
                    x = x(valid);
                    y = y(valid);
                    if numel(x) < 2, continue; end

                    c = struct();
                    c.name = sanitizeName(sprintf('%s_%s_c%d_c%d', book, sheetName, xCol, yCol));
                    c.exportName = c.name;
                    c.book = book;
                    c.sheet = sheetName;
                    c.xCol = xCol;
                    c.yCol = yCol;
                    c.x = x(:);
                    c.y = y(:);
                    c.include = true;
                    c.yRange = max(y) - min(y);
                    c.yStd = std(y);
                    curves(end+1) = c; %#ok<AGROW>
                end
            end
        end
    end

    function curves = readGraphCurves(app)
        curves = struct('name', {}, 'exportName', {}, 'book', {}, 'sheet', {}, 'xCol', {}, 'yCol', {}, ...
            'x', {}, 'y', {}, 'include', {}, 'yRange', {}, 'yStd', {});

        seenDatasets = {};
        emptyGraphStreak = 0;
        for graphIndex = 1:50
            graphName = sprintf('Graph%d', graphIndex);
            try
                existsValue = double(invoke(app, 'Evaluate', sprintf('exist(%s)', graphName)));
            catch
                existsValue = 0;
            end

            if existsValue ~= 3
                emptyGraphStreak = emptyGraphStreak + 1;
                if emptyGraphStreak >= 3, break; end
                continue;
            end
            emptyGraphStreak = 0;

            plotCount = getFirstLayerPlotCount(app, graphName);
            if plotCount < 1
                continue;
            end

            invoke(app, 'Execute', sprintf('win -a %s; layer -s 1;', graphName));
            for plotIndex = 1:plotCount
                yDataset = getLabTalkString(app, sprintf('layer.plot%d.name$', plotIndex));
                if isempty(yDataset) || strcmp(yDataset, sprintf('layer.plot%d.name$', plotIndex))
                    continue;
                end
                if any(strcmp(seenDatasets, yDataset))
                    continue;
                end
                seenDatasets{end+1} = yDataset; %#ok<AGROW>

                xDataset = previousDatasetName(yDataset);
                if isempty(xDataset), continue; end

                [x, y] = readDatasetPair(app, xDataset, yDataset);
                if numel(x) < 2, continue; end

                c = struct();
                c.name = sanitizeName(sprintf('%s_%s', graphName, yDataset));
                c.exportName = c.name;
                c.book = graphName;
                c.sheet = 'Layer1';
                c.xCol = NaN;
                c.yCol = NaN;
                c.x = x(:);
                c.y = y(:);
                c.include = true;
                c.yRange = max(y) - min(y);
                c.yStd = std(y);
                curves(end+1) = c; %#ok<AGROW>
            end
        end
    end

    function plotCount = getFirstLayerPlotCount(app, graphName)
        plotCount = 0;
        try
            graphLayer = invoke(app, 'FindGraphLayer', graphName);
            dataPlots = get(graphLayer, 'DataPlots');
            plotCount = double(get(dataPlots, 'Count'));
        catch
            plotCount = 0;
        end
    end

    function text = getLabTalkString(app, expression)
        invoke(app, 'Execute', sprintf('strtmp$=%s;', expression));
        text = strtrim(char(get(app, 'LTStr', 'strtmp')));
    end

    function [x, y] = readDatasetPair(app, xDataset, yDataset)
        maxRows = 20000;
        missingStreak = 0;
        started = false;
        x = nan(maxRows, 1);
        y = nan(maxRows, 1);
        count = 0;

        for row = 1:maxRows
            xv = readDatasetValue(app, xDataset, row);
            yv = readDatasetValue(app, yDataset, row);

            if isfinite(xv) && isfinite(yv)
                count = count + 1;
                x(count) = xv;
                y(count) = yv;
                missingStreak = 0;
                started = true;
            else
                if started
                    missingStreak = missingStreak + 1;
                    if missingStreak >= 50
                        break;
                    end
                end
            end
        end

        x = x(1:count);
        y = y(1:count);
    end

    function value = readDatasetValue(app, datasetName, row)
        value = double(invoke(app, 'Evaluate', sprintf('%s[%d]', datasetName, row)));
        if abs(value) < 1e-299
            value = NaN;
        end
    end

    function xDataset = previousDatasetName(yDataset)
        tokens = regexp(yDataset, '^(.*_)([A-Z]+)$', 'tokens', 'once');
        if isempty(tokens)
            xDataset = '';
            return;
        end
        prev = previousColumnLetters(tokens{2});
        if isempty(prev)
            xDataset = '';
        else
            xDataset = [tokens{1} prev];
        end
    end

    function prev = previousColumnLetters(letters)
        n = 0;
        for k = 1:numel(letters)
            n = n * 26 + (double(letters(k)) - double('A') + 1);
        end
        n = n - 1;
        if n < 1
            prev = '';
            return;
        end
        prev = '';
        while n > 0
            rem = mod(n - 1, 26);
            prev = [char(double('A') + rem) prev]; %#ok<AGROW>
            n = floor((n - 1) / 26);
        end
    end

    function data = readActiveWorksheetMatrix(app, nRows, nCols)
        nRows = min(nRows, 20000);
        data = nan(nRows, nCols);
        for col = 1:nCols
            for row = 1:nRows
                value = invoke(app, 'Evaluate', sprintf('col(%d)[%d]', col, row));
                value = double(value);
                if abs(value) < 1e-299
                    value = NaN;
                end
                data(row, col) = value;
            end
        end
        keepRows = any(isfinite(data), 2);
        data = data(keepRows, :);
    end

    function refreshCurveTable()
        tableData = cell(numel(state.curves), 6);
        for i = 1:numel(state.curves)
            tableData{i, 1} = state.curves(i).include;
            tableData{i, 2} = state.curves(i).exportName;
            tableData{i, 3} = state.curves(i).name;
            tableData{i, 4} = numel(state.curves(i).x);
            tableData{i, 5} = state.curves(i).yRange;
            tableData{i, 6} = state.curves(i).yStd;
        end
        set(curveTable, 'Data', tableData);
    end

    function tableEdited(~, event)
        row = event.Indices(1);
        col = event.Indices(2);
        if col == 1
            state.curves(row).include = logical(event.NewData);
            plotPreview();
        elseif col == 2
            newName = sanitizeName(char(event.NewData));
            if isempty(newName)
                newName = state.curves(row).name;
            end
            state.curves(row).exportName = newName;
            refreshCurveTable();
        end
    end

    function setAllIncluded(value)
        for i = 1:numel(state.curves)
            state.curves(i).include = value;
        end
        refreshCurveTable();
        plotPreview();
    end

    function scaleControlChanged(~, ~)
        set(multiplyXEdit, 'Enable', ternary(get(multiplyXCheck, 'Value'), 'on', 'off'));
        set(multiplyYEdit, 'Enable', ternary(get(multiplyYCheck, 'Value'), 'on', 'off'));
        try
            plotPreview();
        catch err
            setStatus(['Invalid multiplier: ' err.message]);
        end
    end

    function plotPreview(~, ~)
        xMultiplier = selectedMultiplier(multiplyXCheck, multiplyXEdit, 'x');
        yMultiplier = selectedMultiplier(multiplyYCheck, multiplyYEdit, 'y');
        cla(ax);
        hold(ax, 'on');
        rows = {};
        rowCount = 0;
        for i = 1:numel(state.curves)
            if ~state.curves(i).include, continue; end
            x = state.curves(i).x .* xMultiplier;
            y = state.curves(i).y .* yMultiplier;
            plot(ax, x, y, 'LineWidth', 1.1, 'DisplayName', state.curves(i).name);

            take = min(3, numel(x));
            for k = 1:take
                rowCount = rowCount + 1;
                rows(rowCount, :) = {state.curves(i).name, x(k), y(k)}; %#ok<AGROW>
            end
        end
        hold(ax, 'off');
        grid(ax, 'on');
        xlabel(ax, scaledAxisLabel('x', xMultiplier));
        ylabel(ax, scaledAxisLabel('y', yMultiplier));
        title(ax, 'Selected curves, shared axes');
        if rowCount > 0
            legend(ax, 'Location', 'best');
        end
        set(sampleTable, 'Data', rows);
    end

    function exportSelected(~, ~)
        if isempty(state.curves)
            setStatus('Load curves before exporting.');
            return;
        end
        state.outputDir = strtrim(get(outputEdit, 'String'));
        if isempty(state.outputDir)
            setStatus('Choose an output folder.');
            return;
        end
        if ~isfolder(state.outputDir), mkdir(state.outputDir); end

        count = 0;
        try
            xMultiplier = selectedMultiplier(multiplyXCheck, multiplyXEdit, 'x');
            yMultiplier = selectedMultiplier(multiplyYCheck, multiplyYEdit, 'y');
            for i = 1:numel(state.curves)
                if ~state.curves(i).include, continue; end
                x = state.curves(i).x .* xMultiplier;
                y = state.curves(i).y .* yMultiplier;
                exportName = sanitizeName(state.curves(i).exportName);
                if isempty(exportName)
                    exportName = state.curves(i).name;
                end
                outPath = uniqueOutputPath(state.outputDir, [exportName '.txt']);
                writeCurveFile(outPath, x, y);
                count = count + 1;
            end
            setStatus(sprintf('Exported %d txt files to %s', count, state.outputDir));
        catch err
            setStatus(['Export failed: ' err.message]);
        end
    end

    function closeGui(~, ~)
        closeOriginApp();
        delete(fig);
    end

    function closeOriginApp()
        if ~isempty(state.app)
            try
                invoke(state.app, 'Exit');
            catch
            end
            state.app = [];
        end
    end

    function setStatus(message)
        set(statusText, 'String', message);
        drawnow;
    end
end

function label(parent, text, y)
    uicontrol(parent, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.05 y-0.02 0.30 0.04], 'String', text, ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.98 0.98 0.99], ...
        'ForegroundColor', [0.10 0.12 0.16]);
end

function name = sanitizeName(name)
    name = regexprep(name, '[^\w\-.]+', '_');
    name = regexprep(name, '_+', '_');
    name = regexprep(name, '^_+|_+$', '');
    if isempty(name)
        name = 'curve';
    end
end

function path = uniqueOutputPath(folder, fileName)
    [~, baseName, ext] = fileparts(fileName);
    path = fullfile(folder, [baseName ext]);
    if ~isfile(path)
        return;
    end

    index = 2;
    while true
        path = fullfile(folder, sprintf('%s_%d%s', baseName, index, ext));
        if ~isfile(path)
            return;
        end
        index = index + 1;
    end
end

function multiplier = selectedMultiplier(checkHandle, editHandle, axisName)
    if ~get(checkHandle, 'Value')
        multiplier = 1;
        return;
    end

    textValue = strtrim(get(editHandle, 'String'));
    multiplier = str2double(textValue);
    if isempty(textValue) || ~isscalar(multiplier) || ~isfinite(multiplier)
        error('%s multiplier must be a finite number such as 1000 or 1e12.', axisName);
    end
end

function text = scaledAxisLabel(axisName, multiplier)
    if multiplier == 1
        text = axisName;
    else
        text = sprintf('%s (scaled by %.15g)', axisName, multiplier);
    end
end

function writeCurveFile(filePath, x, y)
    fid = fopen(filePath, 'w');
    if fid < 0
        error('Could not write output file: %s', filePath);
    end
    fileCleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%.17g\t%.17g\n', [double(x(:)), double(y(:))].');
end

function out = ternary(condition, a, b)
    if condition
        out = a;
    else
        out = b;
    end
end
