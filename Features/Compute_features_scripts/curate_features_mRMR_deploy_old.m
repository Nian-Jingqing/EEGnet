function [final_dataset_mRMR,final_feature_labels_mRMR,currFeatures_curated,varargout] = curate_features_mRMR_deploy(Featurefiles_basename, Featurefiles_directory, YY_final, max_features)

% Find curated features:
Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_*.mat']);
currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);

%% Iterative mRMR:                
curr_labels_mRMR = YY_final; % Get current labels for mRMR

wind_dataset_mRMR = []; wind_feature_labels_mRMR = [];
wind_output_features = cell(1,length(currFeatures_curated)); wind_output_scores = cell(1,length(currFeatures_curated));

for i = 1:length(currFeatures_curated) % Number of features
    feat_file = load([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' currFeatures_curated{i} '.mat']);
    
    % Convert linear indices to nested index:
    curr_Feature_curated = cellfun(@(x)reshape(x,[feat_file.Feature_size]),feat_file.Feature,'un',0);
    
    freq_dataset_mRMR = []; freq_feature_labels_mRMR = [];
    freq_output_features = cell(1,feat_file.Feature_size(1)); freq_output_scores = cell(1,feat_file.Feature_size(1));
    
    for m = 1:feat_file.Feature_size(1) % Number of Windows        
    % for m = 1:4 % Number of Windows   
        elec_dataset_mRMR = []; elec_feature_labels_mRMR = [];
        % elec_dataset_mRMR = cell(1,feat_file.Feature_size(2)); elec_feature_labels_mRMR = cell(1,feat_file.Feature_size(2));
        elec_output_features = cell(1,feat_file.Feature_size(2)); elec_output_scores = cell(1,feat_file.Feature_size(2));
        
        for n = 1:feat_file.Feature_size(2) % Number of Frequency Windows
            
            if length(feat_file.Feature_size) == 4 % Single Frequency Computation
                
                % Get current dataset for mRMR:
                curr_feature_size = feat_file.Feature_size(end-1:end);
                curr_dataset_mRMR = cellfun(@(x)squeeze(x(m,n,:,:)),curr_Feature_curated,'un',0); 
                curr_dataset_mRMR = cell2mat(cellfun(@(x)x(:),curr_dataset_mRMR,'un',0))';
                
                % Run mRMR at this level:
                [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR,elec_feature_labels_mRMR] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,elec_dataset_mRMR,elec_feature_labels_mRMR);
                % [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR{n},elec_feature_labels_mRMR{n}] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,elec_dataset_mRMR{n},elec_feature_labels_mRMR{n});

                
                disp(['Running mRMR loop for function ' currFeatures_curated{i} ' Window ' num2str(m) ' FrequencyBand ' num2str(n)]);
               
                
            else % Between Frequency Computation
                
                innerFreq_dataset_mRMR = []; innerFreq_feature_labels_mRMR = [];
                % elec_dataset_mRMR = cell(1,feat_file.Feature_size(2)); elec_feature_labels_mRMR = cell(1,feat_file.Feature_size(2));
                innerFreq_output_features = cell(1,feat_file.Feature_size(3)); innerFreq_output_scores = cell(1,feat_file.Feature_size(3));
                
                
                for p = 1:feat_file.Feature_size(3)
                    % flip p and n index positions because this was not
                    % kept consistent from the single frequency computation
                    % where n is the second index - fix this by indexing
                    % curr_Feature_curated(m,n,p,:,:) in
                    % curate_features_deploy in future
                    
                    % if ~isempty(curr_Feature{1}{m}{p,n}) curr_Feature_curated(m,p,n,:,:) = curr_Feature{1}{m}{p,n}; end    % TO FIX - need to add feat_size to this to make it compatible with features that are not a single value per pair of electrodes

                    % Get current dataset for mRMR:
                    curr_feature_size = feat_file.Feature_size(end-1:end);
                    curr_dataset_mRMR = cellfun(@(x)squeeze(x(m,p,n,:,:)),curr_Feature_curated,'un',0);
                    curr_dataset_mRMR = cell2mat(cellfun(@(x)x(:),curr_dataset_mRMR,'un',0))';
                    
                    % Run mRMR at this level:
                    [innerFreq_output_features{n}, innerFreq_output_scores{n},innerFreq_dataset_mRMR,innerFreq_feature_labels_mRMR] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,innerFreq_dataset_mRMR,innerFreq_feature_labels_mRMR);
                    % [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR{n},elec_feature_labels_mRMR{n}] = mRMR_iterate_loop(curr_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,elec_dataset_mRMR{n},elec_feature_labels_mRMR{n});
                    
                    disp(['Running mRMR loop for function ' currFeatures_curated{i} ' Window ' num2str(m) ' FrequencyBands ' num2str(n) '-' num2str(p)]);

                end
                
                % Run mRMR at this level:
                % elec_dataset_mRMR = cat(2,elec_dataset_mRMR{:});
                % elec_feature_labels_mRMR = cat(2,elec_feature_labels_mRMR{:});
                curr_feature_size = [feat_file.Feature_size(3) size(innerFreq_output_features{1},2)];
                [elec_output_features{n}, elec_output_scores{n},elec_dataset_mRMR,elec_feature_labels_mRMR] = mRMR_iterate_loop(innerFreq_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,elec_dataset_mRMR,elec_feature_labels_mRMR,innerFreq_feature_labels_mRMR);
                
            end
        end
        
        % Run mRMR at this level:
        % elec_dataset_mRMR = cat(2,elec_dataset_mRMR{:});
        % elec_feature_labels_mRMR = cat(2,elec_feature_labels_mRMR{:});
        curr_feature_size = [feat_file.Feature_size(2) size(elec_output_features{1},2)];
        [freq_output_features{m}, freq_output_scores{m},freq_dataset_mRMR,freq_feature_labels_mRMR] = mRMR_iterate_loop(elec_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,freq_dataset_mRMR,freq_feature_labels_mRMR,elec_feature_labels_mRMR);
                
    end
    
    % Run mRMR at this level:
    curr_feature_size = [feat_file.Feature_size(1) size(freq_output_features{m},2)];
    [wind_output_features{i}, wind_output_scores{i},wind_dataset_mRMR,wind_feature_labels_mRMR] = mRMR_iterate_loop(freq_dataset_mRMR,curr_labels_mRMR,curr_feature_size,max_features,freq_dataset_mRMR,wind_feature_labels_mRMR,freq_feature_labels_mRMR);
    
end

final_dataset_mRMR = wind_dataset_mRMR;
final_feature_labels_mRMR = wind_feature_labels_mRMR;

varargout{1} = [];
varargout{1}.wind_output_features = wind_output_features;
varargout{1}.wind_output_scores = wind_output_scores;
varargout{1}.freq_output_features = freq_output_features;
varargout{1}.freq_output_scores = freq_output_scores;
varargout{1}.freq_dataset_mRMR = freq_dataset_mRMR;
varargout{1}.freq_feature_labels_mRMR = freq_feature_labels_mRMR;
varargout{1}.elec_output_features = elec_output_features;
varargout{1}.elec_dataset_mRMR = elec_dataset_mRMR;
varargout{1}.elec_feature_labels_mRMR = elec_feature_labels_mRMR;
if exist('innerFreq_output_features') varargout{1}.innerFreq_output_features = innerFreq_output_features; end
if exist('innerFreq_output_scores') varargout{1}.innerFreq_output_scores = innerFreq_output_scores; end
if exist('innerFreq_dataset_mRMR') varargout{1}.innerFreq_dataset_mRMR = innerFreq_dataset_mRMR; end
if exist('innerFreq_feature_labels_mRMR') varargout{1}.innerFreq_feature_labels_mRMR = innerFreq_feature_labels_mRMR; end

%% Old code:

% Featurefiles_curated_dir = dir([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_*.mat']);
% % currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) str2num(x{2}),currFeatures_curated);
% currFeatures_curated = cellfun(@(x) strsplit(x,{'AllEpochs_','.mat'}),{Featurefiles_curated_dir.name},'un',0); currFeatures_curated = cellfun(@(x) x{2},currFeatures_curated,'un',0);
% Features_to_process = setdiff(Features_to_process,currFeatures_curated);
% % if ~isempty(Featurefiles_dir) && isempty(Featurefiles_curated_dir)
% if ~isempty(Featurefiles_dir) && ~isempty(Features_to_process)
%     Featurefiles_names = {Featurefiles_dir(:).name}; curr_file_order = cell2mat(cellfun(@(x) str2num(x((strfind(x,'h') + 1):(strfind(x,'.') - 1))),Featurefiles_names,'UniformOutput',0));
%     [~, curr_file_order_sorted_idx] = sort(curr_file_order);
%     parfor i = 1:length(Features_to_process) % This is the PARFOR 
%         Feature = []; Feature_labels = [];
%         for j = 1:length(curr_file_order_sorted_idx)
%             
%             curr_Epoch = curr_file_order_sorted_idx(j);
%             disp(['Running Feature ' Features_to_process{i} ' Epoch ' num2str(curr_Epoch)]);
%             curr_file = load([Featurefiles_directory filesep Featurefiles_dir(curr_Epoch).name]);
%             
%             % Compute derivative features from original features if not included in saved file:
%             if ~isfield(curr_file,Features_to_process{i})
%                 if strcmp(Features_to_process{i},'CFC_SI_mag') || strcmp(Features_to_process{i},'CFC_SI_theta')
%                     for m = 1:length(curr_file.CFC_SI{1}) % Number of Windows
%                         for n = 1:size(curr_file.CFC_SI{1}{1},2) % Number of Frequency Windows - columns
%                             for p = 1:size(curr_file.CFC_SI{1}{1},1) % Number of Frequency Windows 2 - rows
%                                 if ~isempty(curr_file.CFC_SI{1}{m}{p,n})
%                                     num_chan = size(curr_file.CFC_SI{1}{m}{p,n},1);
%                                     for ch1 = 1:num_chan                                        
%                                         curr_file.CFC_SI_mag{1}{m}{p,n}(ch1,:) = abs(curr_file.CFC_SI{1}{m}{p,n}(ch1,:));
%                                         curr_file.CFC_SI_theta{1}{m}{p,n}(ch1,:) = angle(curr_file.CFC_SI{1}{m}{p,n}(ch1,:));
%                                     end
%                                 elseif p == size(curr_file.CFC_SI{1}{1},1)
%                                     curr_file.CFC_SI_mag{1}{m}{p,n} = [];
%                                 end
%                             end
%                         end
%                     end
%                 end
%             end
% 
%             
%             if isfield(curr_file,'analyzedData')
%                 curr_file_cell = struct2cell(curr_file.analyzedData); curr_file_cell_var = fields(curr_file.analyzedData);
%             else
%                 curr_file_cell = struct2cell(curr_file); curr_file_cell_var = fields(curr_file);
%             end
%             curr_feature_idx = cell2mat(cellfun(@(x) strcmp(x,Features_to_process{i}), curr_file_cell_var,'UniformOutput',0));
%             curr_Feature = curr_file_cell{curr_feature_idx};
%             
%             % Give feature output as cell or as a matrix:
%             if cell_output
%                 Feature = [Feature; curr_Feature];
%             else
%                 if size(curr_Feature{1}{1},1) == 1 % Single Frequency Computation
%                     curr_Feature_curated = NaN([length(curr_Feature{1}),length(curr_Feature{1}{1}),size(curr_Feature{1}{1}{1})]);
%                     curr_feature_label_curated = cell([length(curr_Feature{1}),length(curr_Feature{1}{1}),size(curr_Feature{1}{1}{1})]);
%                     
%                 else % Between Frequency Computation
%                     feat_size = [];
%                     for k = 1:length(size(curr_Feature{1}{1}{1}))
%                         feat_size{k} = max(max(cell2mat(cellfun(@(x) size(x,k),curr_Feature{1}{1},'UniformOutput',0))));                        
%                     end
%                     feat_size = cell2mat(feat_size);
%                     curr_Feature_curated = NaN([length(curr_Feature{1}),size(curr_Feature{1}{1}),feat_size]);
%                     curr_Feature_label_curated = NaN([length(curr_Feature{1}),size(curr_Feature{1}{1}),feat_size]);
%                 end
%                     
%                 for m = 1:length(curr_Feature{1}) % Number of Windows
%                     curr_feature_label = '';
%                     curr_feature_label = [curr_feature_label 'W' num2str(m)];
%                     for n = 1:size(curr_Feature{1}{1},2) % Number of Frequency Windows
%                         curr_feature_label = [curr_feature_label 'F' num2str(n)];
%                         if size(curr_Feature{1}{m},1) == 1 % Single Frequency Computation
%                             curr_Feature_curated(m,n,:,:) = curr_Feature{1}{m}{n};
%                         else % Between Frequency Computation
%                             for p = 1:size(curr_Feature{1}{1},1)
%                                 if ~isempty(curr_Feature{1}{m}{p,n}) curr_Feature_curated(m,p,n,:,:) = curr_Feature{1}{m}{p,n}; end    % TO FIX - need to add feat_size to this to make it compatible with features that are not a single value per pair of electrodes 
%                             end
%                         end
%                     end
%                 end
%                 Feature{j} = curr_Feature_curated(:);
%             end
%         end 
%         analyzedData = []; analyzedData.Feature = Feature; analyzedData.Feature_size = size(curr_Feature_curated); analyzedData.curr_Feature_curated = curr_Feature_curated;
%         parsave_struct([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' Features_to_process{i}], analyzedData, 0)
%         % parsave([Featurefiles_directory filesep Featurefiles_basename '_AllEpochs_' Features_to_process{i}],Feature,Feature_size);
%     end
%     features_curated = true;
% elseif isempty(Featurefiles_dir)
%     disp('\n Features not computed yet \n');
% else
%     disp('\n Features already curated \n');
%     features_curated = true;
% end