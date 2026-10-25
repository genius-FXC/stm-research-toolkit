function outputFiles = saveKondoFitResult(result, filePath, outputDir)
%SAVEKONDOFITRESULT Save xyfit, full fit data, and fit parameters.

    if nargin < 2 || isempty(filePath)
        filePath = result.filePath;
    end
    if nargin < 3 || isempty(outputDir)
        outputDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'output');
    end
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    [~, baseName] = fileparts(filePath);
    stamp = datestr(now, 'yyyymmdd_HHMMSS');

    xyfitPath = fullfile(outputDir, [baseName '_kondo_fit_' stamp '_xyfit.txt']);
    fullPath = fullfile(outputDir, [baseName '_kondo_fit_' stamp '_full.csv']);
    paramsPath = fullfile(outputDir, [baseName '_kondo_fit_' stamp '_params.txt']);

    fid = fopen(xyfitPath, 'w');
    if fid < 0
        error('saveKondoFitResult:WriteFailed', 'Could not write output file: %s', xyfitPath);
    end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'gamma: %.15g\n', result.Gamma);
    fprintf(fid, 'x\tyfit\n');
    fprintf(fid, '%.15g\t%.15g\n', [result.x(:), result.yfit(:)].');
    clear cleanup;

    fullTable = table(result.x(:), result.y(:), result.yfit(:), result.fitNormalized(:), ...
        'VariableNames', {'x', 'y', 'yfit', 'fitNormalized'});
    writetable(fullTable, fullPath);

    fid = fopen(paramsPath, 'w');
    if fid < 0
        error('saveKondoFitResult:WriteFailed', 'Could not write output file: %s', paramsPath);
    end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'sourceFile: %s\n', result.filePath);
    fprintf(fid, 'Gamma: %.15g\n', result.Gamma);
    fprintf(fid, 'phi: %.15g\n', result.phi);
    fprintf(fid, 'a: %.15g\n', result.a);
    fprintf(fid, 'biasOffset: %.15g\n', result.biasOffset);
    fprintf(fid, 'bgSlope: %.15g\n', result.bgSlope);
    fprintf(fid, 'bgIntercept: %.15g\n', result.bgIntercept);
    fprintf(fid, 'resnormNormalized: %.15g\n', result.resnormNormalized);
    fprintf(fid, 'rmseNormalized: %.15g\n', result.rmseNormalized);
    fprintf(fid, 'temperature: %.15g\n', result.temperature);
    fprintf(fid, 'lockInAmplitude: %.15g\n', result.lockInAmplitude);
    fprintf(fid, 'useLockIn: %d\n', result.useLockIn);
    fprintf(fid, 'units: %s\n', result.units);
    fprintf(fid, 'model: %s\n', result.model);
    fprintf(fid, 'biasOffsetInitial: %.15g\n', result.biasOffsetInitial);
    fprintf(fid, 'biasOffsetLower: %.15g\n', result.biasOffsetLower);
    fprintf(fid, 'biasOffsetUpper: %.15g\n', result.biasOffsetUpper);
    fprintf(fid, 'fixBiasOffset: %d\n', result.fixBiasOffset);
    fprintf(fid, 'gammaInitial: %.15g\n', result.gammaInitial);
    fprintf(fid, 'gammaLower: %.15g\n', result.gammaLower);
    fprintf(fid, 'gammaUpper: %.15g\n', result.gammaUpper);
    fprintf(fid, 'yMaxUsedForDenormalization: %.15g\n', result.yMax);
    clear cleanup;

    outputFiles = struct('xyfit', xyfitPath, 'full', fullPath, 'params', paramsPath);
end
