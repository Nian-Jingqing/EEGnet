%% function [ ] = mRMR_iterate( )

curr_path = 'Z:\research_code\EEG_fMRI_Modelling\Joint_Feature_Learning\Classification';
% curr_path = pwd;
cd(curr_path)
cd ..
base_path = pwd;
cd(curr_path)
% isdir_present = isempty(dir('MRMR Input Matrix'));
% if isdir_present mkdir([curr_path filesep 'MRMR Input Matrix']); end
%% Run from here for each subject:

subject_num = 2;
session_num = 2;
max_features = 10000;

windowLength = [100,200,400,1000];
windowStep = [50, 100, 200, 500];
numTimePoints = 5000;

% For Saurabh Testing:
% files_dir = dir([base_path filesep 'Main' filesep 'EEG_Results' filesep 'Rev_Sub' num2str(subject_num) '_Ses' num2str(session_num) '_Epoch*.mat']);

% For Tyler:
files_dir_path = 'E:\Research_data\Analyzed_data\EEG_fMRI_Combined_Dataset_Features';
% files_dir_path = 'Z:\research_code\EEG_fMRI_Modelling\Joint_Feature_Learning\Main\EEG_Results';
files_dir = dir([files_dir_path filesep 'Rev_Sub' num2str(subject_num) '_Ses' num2str(session_num) '_Epoch*.mat']);
cd(files_dir_path);
% load('Crash_workspace.mat', 'files_dir_matfile');
files_dir_matfile = cellfun(@(x) matfile(x),{files_dir.name},'UniformOutput',false);
cd(curr_path);
% files_dir_matfile = cellfun(@(x) matfile(x),{[files_dir.folder filesep files_dir.name]},'UniformOutput',false);

%% Load first Epoch to get sizes for everything:
dataset = load([files_dir(1).folder filesep files_dir(1).name]); dataset = dataset.analyzedData;
numFeatures = length(dataset);
numWindowLengths = length(dataset{1});
numWindows = cellfun(@length,dataset{1}(:));
numFreqBands = length(dataset{1}{1}{1});

% Load output labels and arrange according to files_dir:
load(['Z:\research_code\EEG_fMRI_Modelling\Deep_Learning\Dataset' filesep 'EEGfMRI_data_Subject' num2str(subject_num) '_Session' num2str(session_num) '.mat'], 'fMRI_labels_selected_window_thresh');
curr_file_order = cell2mat(cellfun(@(x) str2num(x((strfind(x,'h') + 1):(strfind(x,'.') - 1))),{files_dir.name},'UniformOutput',false));
mat_labels = fMRI_labels_selected_window_thresh(curr_file_order,:);

% Modify output labels from multilabel to singlelabel (Find better way than this):
curr_labels_mRMR = zeros(size(mat_labels,1),1);
for i = 1:size(mat_labels,1)
    curr_values = find(mat_labels(i,:)); 
    if isempty(curr_values) curr_values = 0; end
    curr_labels_mRMR(i) = curr_values(1);    % Find a better way to do this for multi-label classification
end


%% Iterate over all the features:
i_output_features = cell(1,numFeatures);
i_output_scores = cell(1,numFeatures);
i_dataset_mRMR = [];
i_feature_labels_mRMR = [];
% for i = 6:numFeatures
for i = [1 3 4 5 6 7 8 9 10 11]
    j_output_features = cell(1,numWindowLengths);
    j_output_scores = cell(1,numWindowLengths);
    j_dataset_mRMR = [];
    j_feature_labels_mRMR = [];
    
    % Iterate over all window lengths:
    for j = 1:numWindowLengths
        
        k_output_features = cell(1,numWindows(j));
        k_output_scores = cell(1,numWindows(j));
        k_dataset_mRMR = [];
        k_feature_labels_mRMR = [];
        
        % Iterate over all the windows:
        for k = 1:numWindows(j) 
            
            disp(['Running i ' num2str(i) ' j ' num2str(j) ' k ' num2str(k)])
            
            curr_dataset = cell(1,length(files_dir));
            curr_feature_labels = load([curr_path filesep 'Feature_Labels' filesep 'Feature_Labels_' num2str(i) '_' num2str(j) '_' num2str(k)]);
            curr_feature_labels = curr_feature_labels.analyzedData;
            
            num_feature_dims = length(size(curr_feature_labels));
            
            % Iterate over all the epochs:
            parfor m = 1:length(files_dir)
                currFile = files_dir(m).name;
                [~,firstFile_name] = fileparts(currFile);

                % Load the corresponding file and save in curr_dataset:
                % disp(['Currently running ' currFile ])                
                % dataset = load(currFile);                
                % curr_dataset{m} = dataset.analyzedData{i}{j}{k};
                dataset = files_dir_matfile{m}.analyzedData(1,i);
                % dataset = dataset_matfile.analyzedData(1,i);
                curr_dataset{m} = dataset{1}{j}{k};
            end
            
            % Feed the collected features into mRMR:
            permute_mat = circshift(1:num_feature_dims,1);
            if iscell(curr_dataset{1})
                curr_dataset_temp = cellfun(@(x) permute(cat(num_feature_dims,x{:}),permute_mat),curr_dataset,'UniformOutput',false);
            else
                curr_dataset_temp = cellfun(@(x) cat(num_feature_dims,x(:)),curr_dataset,'UniformOutput',false);
            end
            curr_dataset = cellfun(@(x) x(:)',curr_dataset_temp,'UniformOutput',false);
            
            % Check for same sized columns in curr_dataset:
            curr_dataset_len = cell2mat(cellfun(@(x)length(x),curr_dataset,'UniformOutput',false));
            curr_dataset_len_unique = unique(curr_dataset_len);
            if length(curr_dataset_len_unique) > 1
                
                % Find location/s that have shorter size cells:
                curr_dataset_num = zeros(1,length(curr_dataset_len_unique));
                for kk = 1:length(curr_dataset_len_unique)
                    curr_dataset_temp = curr_dataset_len == curr_dataset_len_unique(kk);
                    curr_dataset_num(kk) = sum(curr_dataset_temp);               
                end
                [~,curr_dataset_num_max_idx] = max(curr_dataset_num);
                curr_dataset_len_unique_idx = find(curr_dataset_len ~= curr_dataset_len_unique(curr_dataset_num_max_idx));
                
                % Zero pad locations that have shorter cell lengths:
                curr_dataset_temp = curr_dataset;
                for jj = 1:length(curr_dataset_len_unique_idx)
                    pad_amount = curr_dataset_len_unique(curr_dataset_num_max_idx) - curr_dataset_len(curr_dataset_len_unique_idx(jj));
                    curr_dataset_temp{curr_dataset_len_unique_idx(jj)} = padarray(cell2mat(curr_dataset_temp(curr_dataset_len_unique_idx(jj))),[0 pad_amount],NaN,'post');
                end
                curr_dataset = curr_dataset_temp;
            end
            
            % Continue with mRMR:
            curr_dataset_mRMR = cat(1,curr_dataset{:}); curr_dataset_mRMR(isnan(curr_dataset_mRMR)) = 0;
            curr_feature_labels_mRMR = {curr_feature_labels{:}};
            
            [k_output_features{k}, k_output_scores{k}] = mRMR(curr_dataset_mRMR,curr_labels_mRMR,min(floor(max_features/2),size(curr_dataset_mRMR,2)));
            
            k_dataset_mRMR = cat(2,k_dataset_mRMR,curr_dataset_mRMR(:,k_output_features{k}));
            k_feature_labels_mRMR = cat(2,k_feature_labels_mRMR,curr_feature_labels_mRMR(k_output_features{k}));
            
            % disp(['Finished i ' num2str(i) ' j ' num2str(j) ' k ' num2str(k)])
            
        end
       
        [j_output_features{j}, j_output_scores{j}] = mRMR(k_dataset_mRMR,curr_labels_mRMR,min(floor(max_features/2),size(k_dataset_mRMR,2)));
        
        j_dataset_mRMR = cat(2,j_dataset_mRMR,k_dataset_mRMR(:,j_output_features{j}));
        j_feature_labels_mRMR = cat(2,j_feature_labels_mRMR,k_feature_labels_mRMR(j_output_features{j}));
        
        save(['mRMR_output_features_Sub' num2str(subject_num) '_Ses' num2str(session_num) '_i' num2str(i) '_j' num2str(j)],'curr_labels_mRMR',...
            'k_output_features','k_output_scores','k_dataset_mRMR','k_feature_labels_mRMR',...
            'j_output_features','j_output_scores','j_dataset_mRMR','j_feature_labels_mRMR');
    end
    
    [i_output_features{i}, i_output_scores{i}] = mRMR(j_dataset_mRMR,curr_labels_mRMR,min(floor(max_features/2),size(j_dataset_mRMR,2)));
        
    i_dataset_mRMR = cat(2,i_dataset_mRMR,j_dataset_mRMR(:,i_output_features{i}));
    i_feature_labels_mRMR = cat(2,i_feature_labels_mRMR,j_feature_labels_mRMR(i_output_features{i}));
    
    save(['mRMR_output_features_Sub' num2str(subject_num) '_Ses' num2str(session_num) '_i' num2str(i)],'curr_labels_mRMR',...
            'i_output_features','i_output_scores','i_dataset_mRMR','i_feature_labels_mRMR');
end

%% Final mRMR:

% subsess_dataset_mRMR = [];
% subsess_feature_labels_mRMR = [];
% 
% [subsess_output_features{i}, subsess_output_scores{i}] = mRMR(i_dataset_mRMR,curr_labels_mRMR,min(floor(max_features/2),size(i_dataset_mRMR,2)));
%         
% subsess_dataset_mRMR = cat(2,subsess_dataset_mRMR,i_dataset_mRMR(:,subsess_output_features{i}));
% subsess_feature_labels_mRMR = cat(2,subsess_feature_labels_mRMR,i_feature_labels_mRMR(subsess_output_features{i}));
% 
% save(['mRMR_output_features_Sub' num2str(subject_num) '_Ses' num2str(session_num)],'curr_labels_mRMR',...
%     'subsess_output_features','subsess_output_scores','subsess_dataset_mRMR','subsess_feature_labels_mRMR');
