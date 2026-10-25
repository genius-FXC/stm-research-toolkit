function result = runKondoFit(filePath, opts)
%RUNKONDOFIT Read STS data, fit the Kondo resonance, and export fit curves.
%
% result = runKondoFit(filePath, opts)
%
% Required opts fields:
%   xColumn, yColumn, temperature, lockInAmplitude, useLockIn
%
% Optional opts fields:
%   fitMin, fitMax, units, outputDir, closeClassFigure, model, saveOutput
%   biasOffsetInitial, biasOffsetLower, biasOffsetUpper, fixBiasOffset
%   gammaInitial, gammaLower, gammaUpper

    if nargin < 2 || isempty(opts)
        opts = struct();
    end
    opts = withDefaults(opts);

    ensureKondoClassOnPath();

    if ~isfile(filePath)
        error('runKondoFit:MissingFile', 'Data file does not exist: %s', filePath);
    end

    data = readmatrix(filePath);
    if isempty(data) || ~isnumeric(data)
        error('runKondoFit:BadData', 'Could not read numeric matrix from the data file.');
    end

    if opts.xColumn < 1 || opts.yColumn < 1 || ...
            opts.xColumn > size(data, 2) || opts.yColumn > size(data, 2)
        error('runKondoFit:BadColumns', 'Selected x/y columns are outside the data matrix.');
    end

    xAll = data(:, opts.xColumn);
    yAll = data(:, opts.yColumn);
    valid = isfinite(xAll) & isfinite(yAll);
    xAll = xAll(valid);
    yAll = yAll(valid);

    if numel(xAll) < 6
        error('runKondoFit:TooFewPoints', 'At least 6 finite x/y points are required.');
    end

    [xAll, order] = sort(xAll(:));
    yAll = yAll(order);

    rangeMask = true(size(xAll));
    if ~isempty(opts.fitMin) && isfinite(opts.fitMin)
        rangeMask = rangeMask & xAll >= opts.fitMin;
    end
    if ~isempty(opts.fitMax) && isfinite(opts.fitMax)
        rangeMask = rangeMask & xAll <= opts.fitMax;
    end

    x = xAll(rangeMask);
    y = yAll(rangeMask);
    if numel(x) < 6
        error('runKondoFit:TooFewRangePoints', 'The selected fitting range contains fewer than 6 points.');
    end

    yMax = max(y);
    if ~isfinite(yMax) || yMax == 0
        error('runKondoFit:BadYScale', 'The selected y data cannot be normalized because max(y) is zero or invalid.');
    end

    figsBefore = findall(0, 'Type', 'figure');
    h = hurwitzKondoClass(x, y, opts.temperature);
    h.li = opts.lockInAmplitude;
    h.useLi = opts.useLockIn;
    h.units = opts.units;
    h.model = opts.model;
    h.biasOffset = opts.biasOffsetInitial;
    h.Gamma = opts.gammaInitial;
    h.fitLowerBounds = [0, opts.gammaLower/2.542, 0, opts.biasOffsetLower, -Inf, -Inf];
    h.fitUpperBounds = [Inf, opts.gammaUpper/2.542, 2*pi, opts.biasOffsetUpper, Inf, Inf];
    h.fitFixedMask = [false, false, false, false, false, false];
    if opts.fixBiasOffset
        h.fitFixedMask(4) = true;
        h.fitLowerBounds(4) = opts.biasOffsetInitial;
        h.fitUpperBounds(4) = opts.biasOffsetInitial;
    end
    h = h.dofit();

    if opts.closeClassFigure
        figsAfter = findall(0, 'Type', 'figure');
        newFigs = setdiff(figsAfter, figsBefore);
        close(newFigs);
    end

    if isempty(h.fit)
        error('runKondoFit:EmptyFit', 'The Kondo fit did not return h.fit.');
    end

    fitNormalized = h.fit(:);
    yfit = fitNormalized .* yMax;
    residualNormalized = fitNormalized - y(:)./yMax;

    result = struct();
    result.filePath = filePath;
    result.sourceData = data;
    result.xAll = xAll;
    result.yAll = yAll;
    result.x = x(:);
    result.y = y(:);
    result.fitNormalized = fitNormalized;
    result.yfit = yfit;
    result.residualNormalized = residualNormalized;
    result.resnormNormalized = sum(residualNormalized.^2);
    result.rmseNormalized = sqrt(mean(residualNormalized.^2));
    result.yMax = yMax;
    result.h = h;
    result.Gamma = h.Gamma;
    result.phi = h.phi;
    result.a = h.a;
    result.biasOffset = h.biasOffset;
    result.bgSlope = h.bgSlope;
    result.bgIntercept = h.bgIntercept;
    result.temperature = opts.temperature;
    result.lockInAmplitude = opts.lockInAmplitude;
    result.useLockIn = opts.useLockIn;
    result.units = opts.units;
    result.model = opts.model;
    result.biasOffsetInitial = opts.biasOffsetInitial;
    result.biasOffsetLower = opts.biasOffsetLower;
    result.biasOffsetUpper = opts.biasOffsetUpper;
    result.fixBiasOffset = opts.fixBiasOffset;
    result.gammaInitial = opts.gammaInitial;
    result.gammaLower = opts.gammaLower;
    result.gammaUpper = opts.gammaUpper;

    result.outputFiles = struct('xyfit', '', 'full', '', 'params', '');
    if opts.saveOutput
        result.outputFiles = saveKondoFitResult(result, filePath, opts.outputDir);
    end
end

function opts = withDefaults(opts)
    defaults = struct( ...
        'xColumn', 1, ...
        'yColumn', 2, ...
        'temperature', 4, ...
        'lockInAmplitude', 1, ...
        'useLockIn', true, ...
        'fitMin', [], ...
        'fitMax', [], ...
        'units', 'mV', ...
        'outputDir', '', ...
        'closeClassFigure', true, ...
        'saveOutput', false, ...
        'model', 'hurwitz', ...
        'biasOffsetInitial', 0, ...
        'biasOffsetLower', -1, ...
        'biasOffsetUpper', 1, ...
        'fixBiasOffset', false, ...
        'gammaInitial', 1, ...
        'gammaLower', 0, ...
        'gammaUpper', 25.42);

    names = fieldnames(defaults);
    for k = 1:numel(names)
        name = names{k};
        if ~isfield(opts, name) || isempty(opts.(name))
            opts.(name) = defaults.(name);
        end
    end

    opts.xColumn = validateScalarNumber(opts.xColumn, 'xColumn', true);
    opts.yColumn = validateScalarNumber(opts.yColumn, 'yColumn', true);
    opts.temperature = validateScalarNumber(opts.temperature, 'temperature', false);
    opts.lockInAmplitude = validateScalarNumber(opts.lockInAmplitude, 'lockInAmplitude', false);
    opts.biasOffsetInitial = validateScalarNumber(opts.biasOffsetInitial, 'biasOffsetInitial', false);
    opts.biasOffsetLower = validateScalarNumber(opts.biasOffsetLower, 'biasOffsetLower', false);
    opts.biasOffsetUpper = validateScalarNumber(opts.biasOffsetUpper, 'biasOffsetUpper', false);
    opts.gammaInitial = validateScalarNumber(opts.gammaInitial, 'gammaInitial', false);
    opts.gammaLower = validateScalarNumber(opts.gammaLower, 'gammaLower', false);
    opts.gammaUpper = validateScalarNumber(opts.gammaUpper, 'gammaUpper', false);
    opts.useLockIn = logical(opts.useLockIn);
    opts.closeClassFigure = logical(opts.closeClassFigure);
    opts.saveOutput = logical(opts.saveOutput);
    opts.fixBiasOffset = logical(opts.fixBiasOffset);
    if ~(ischar(opts.units) || isstring(opts.units))
        error('runKondoFit:BadUnits', 'units must be mV, V, or uV.');
    end
    opts.units = char(opts.units);
    if ~ismember(opts.units, {'mV', 'V', 'uV'})
        error('runKondoFit:BadUnits', 'units must be mV, V, or uV.');
    end
    if ~(ischar(opts.model) || isstring(opts.model))
        error('runKondoFit:BadModel', 'model must be hurwitz or frota.');
    end
    opts.model = char(opts.model);
    if ~ismember(opts.model, {'hurwitz', 'frota'})
        error('runKondoFit:BadModel', 'model must be hurwitz or frota.');
    end
    if opts.biasOffsetLower > opts.biasOffsetUpper
        error('runKondoFit:BadBounds', 'biasOffsetLower must be less than or equal to biasOffsetUpper.');
    end
    if opts.gammaLower > opts.gammaUpper || opts.gammaLower < 0
        error('runKondoFit:BadBounds', 'Gamma bounds must satisfy 0 <= lower <= upper.');
    end
    if opts.gammaInitial < opts.gammaLower || opts.gammaInitial > opts.gammaUpper
        error('runKondoFit:BadBounds', 'Gamma initial value must lie within the Gamma bounds.');
    end
    if ~opts.fixBiasOffset && ...
            (opts.biasOffsetInitial < opts.biasOffsetLower || opts.biasOffsetInitial > opts.biasOffsetUpper)
        error('runKondoFit:BadBounds', 'biasOffset initial value must lie within the biasOffset bounds.');
    end
end

function value = validateScalarNumber(value, name, mustBeInteger)
    if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value)
        error('runKondoFit:BadOption', '%s must be a finite numeric scalar.', name);
    end
    if mustBeInteger && (value < 1 || round(value) ~= value)
        error('runKondoFit:BadOption', '%s must be a positive integer.', name);
    end
end

function ensureKondoClassOnPath()
    thisDir = fileparts(mfilename('fullpath'));
    demoDir = fullfile(fileparts(thisDir), 'demo');

    if isfolder(demoDir)
        addpath(demoDir, '-end');
    end
    if isfile(fullfile(thisDir, 'hurwitzKondoClass.m'))
        addpath(thisDir, '-begin');
    end

    if exist('hurwitzKondoClass', 'class') ~= 8
        error('runKondoFit:MissingClass', 'Could not find hurwitzKondoClass.m on the Matlab path.');
    end
end
