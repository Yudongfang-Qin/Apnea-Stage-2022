clear all
close all
clc

% define a common frequency (readings per second)
COMMON_FREQUENCY = 10;

% define the sample length
SAMPLE_LENGTH_IN_MINUTES = 10;


% specify the EDF file to be read
% all EDF files MUST follow this convention: if the patient is having the
% condition the file starts with 1_ to 4_ for COPD stages, otherwise if the patient is a control/
% healthy patient, then the file starts with 0_

%%add labels (apnea index)
[allFileInfo,rawData] = readAllCSV("test");
dataY=readtable('Y.csv');
%%
Y_repeat=[];
 % this is the matrix that holds the results
% X1 = zeros(428,540000);
% X = zeros(1,540000);
X1=[];
for index = 2
    fprintf('Processing file %d', index);
    fprintf(".\n");
    data = rawData(index).data; 
    X = []
    name = allFileInfo(index).name
    y0 = cell2mat(cellfun(@(x)isequal(name,x),dataY.file,'UniformOutput', false));
    n = find(y0==1);
    y = dataY.Y(n);
    if length(y) == 2
        Y_repeat = [Y_repeat name ]
        continue
    end

    %read csv files. 

    % data(:,7) = [];
    data(:,6) = [];
    signals = { 'Flow', 'Pulsewave', 'Thorax','Abdomen', 'Snoring' ,'EDFAnnotations'};
    %data = unifiedData(rawData);
    data=transpose(data);
    %delete outliers

    outliers = [NaN, NaN;  NaN, NaN;  NaN, 260;  NaN, 300; NaN, 260; nan, nan];
    indexes_to_delete = [];
    for row=1:size(outliers, 1)
        min_cutoff = outliers(row, 1);
        max_cutoff = outliers(row, 2);

        if ~isnan(min_cutoff)
            indexes_to_delete = [indexes_to_delete find(data(row, :) < min_cutoff)];
        end

        if ~isnan(max_cutoff)
            indexes_to_delete = [indexes_to_delete find(data(row, :) > max_cutoff)];
        end
    end

    indexes_to_delete = unique(indexes_to_delete);
    data(:, indexes_to_delete) = [];
    %rescale between 0 and 1
    disp('Rescaling data...');
    rowmin = min(data, [], 2);
    rowmax = max(data, [], 2);
    L_BOUND = 0;
    U_BOUND = 1;
    data_scaled = rescale(data, L_BOUND, U_BOUND, 'InputMin',rowmin,'InputMax',rowmax);

    disp('Collecting samples...');

    sample_length = 60 * SAMPLE_LENGTH_IN_MINUTES * COMMON_FREQUENCY;

%     sample_points = int32(linspace(1, size(data_scaled, 2), floor(size(data_scaled, 2)/sample_length)+1));
    sample_points = [0:6000:size(data_scaled, 2)];

    % adjust for non-overlapping iteration (todo)
    if sample_points(1) == 1
        sample_points(1) = 0;
    end

    diff = setdiff(0:size(data_scaled, 2)-sample_length, sample_points);

    % how many runs
    runs = length(sample_points)-1;

    signals = size(data, 1);
   

    fprintf('Running with %d samples in total',...
        runs);
    fprintf(".\n");


    %%
    % running the 'span' samples

    for i = 1:runs
         if size(X,1)==15
           continue
         end 
        
%         v = zeros(1, (signals)^2);
        offset_start = sample_points(i)+1;
        offset_end = sample_points(i+1);
%         fprintf('Processing span sample %d/%d with offset: %d - %d\n', i, runs, offset_start, offset_end);
        % subsample 

        subsample = transpose(data_scaled(:,offset_start:offset_end)); %6000*6
        % if all values in the same columns remain the same: 
            %continue. 
        
        idx = subsample(:,max(subsample)==min(subsample));
        if ~isempty(idx)
            continue
        end
        v = reshape(subsample,1,[]);

%         % run Gaurav's algorithm
%         [Aout, B, order, u, relErr] = ...
%             modelEst('sensInd', 1:signals, 'numInp', floor(signals/2),...
%             'data', subsample, 'silentFlag', 0);
%         v= reshape(Aout', 1, [])
%         % make a row-vector out of the result and add it to the main dataset
                 
       X = [X;v]; 
    end
    X1 = [X1;X];
%     X(:,all(X==0,2))=[]

    % add the y-label (todo) X^delta(t)=A*X(t-1)+B*u(t)
%     label=y*ones(runs, 1);
%     label(all(X==0,2),:)=[]
%     X(all(X==0,2),:)=[]
%     X=[X, label];
        
end

% save the dataset as CSV
form = sprintf('%s/%s.csv',"samples",'dataset_baseline')
% csvwrite(form,X1)
csvwrite(form,X1)