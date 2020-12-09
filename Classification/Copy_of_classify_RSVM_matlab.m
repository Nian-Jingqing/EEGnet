function [TrainAccuracy, TestAccuracy, MdlSVM, perfcurve_stats] = classify_RSVM_matlab(X,Y,kernel,testTrainPartition)

% kernel can be 'RBF', or 'linear', or 'polynomial'
% returns accuracy, AUCsvm, specificity and sensitivity

%% Create Testing and Training Partitions:
% tabulate(Y); % Gives the distribution of the classes in Y
CVP = cvpartition(Y,'holdout',1-testTrainPartition);
idxTrain = training(CVP);           % Training-set indices
idxTest = test(CVP);                % Test-set indices

%% Feature Scaling:
X = (X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));

%% Feature Selection:

%% Train an SVM Model:
% MdlSVM = fitrsvm(X(idxTrain,:),Y(idxTrain),'Standardize',true,...
%     'KernelFunction',kernel,'KFold',5,'KernelScale','auto','FitPosterior',1);
try
    MdlSVM = fitrsvm(X(idxTrain,:),Y(idxTrain),'Standardize',true,...
        'KernelFunction',kernel,'KernelScale','auto','FitPosterior',1);
catch e
    if strcmp(e.message,'FitPosterior is not a valid parameter name.')
        MdlSVM = fitrsvm(X(idxTrain,:),Y(idxTrain),'Standardize',true,...
            'KernelFunction',kernel,'KernelScale','auto');
    end
end

% Test the SVM Model:
% [YTrainhat,YTrainhat_NegLoss,YTrainhat_score,YTrainhat_posterior] = resubPredict(MdlSVM);
% [YTesthat,YTesthat_score] = predict(MdlSVM,X(idxTest,:));

%% Accumulate results:
% TrainAccuracy = resubLoss(MdlSVM);
% TestAccuracy = kfoldLoss(MdlSVM);
TrainAccuracy = loss(MdlSVM,X(idxTrain,:),Y(idxTrain));
TestAccuracy = loss(MdlSVM,X(idxTest,:),Y(idxTest));
perfcurve_stats = cell(1,5);
% % [Xsvm,Ysvm,Tsvm,AUCsvm,OPTROCPT] = perfcurve(MdlSVM.Y,(score(:,2)),1);
% figure; plot(Xsvm, Ysvm); hold on; plot(OPTROCPT(1),OPTROCPT(2),'ro'); xlabel('False positive rate'); ylabel('True positive rate'); title('ROC for Matlab SVM Classification');
% specificity = 1-OPTROCPT(1); % Specificity = 1 - FalsePositiveRate
% sensitivity = OPTROCPT(2); % Sensitivity = TruePositiveRate