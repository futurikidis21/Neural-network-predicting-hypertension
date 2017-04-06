%% High blood pressure and socio-demographic profiling: 
% A comparative analysis of Support Vector Machines versus 
% Back-propagation Neural Networks in predicting 
% high blood pressure based on data recorded in 
% the English Longitudinal Study of Ageing

%% Set working directory for the analysis
cd 'C:\Users\Render02\Desktop\MSc Data Science\Neural networks\Coursework'

%% Set filename and read the excel data-file
filename = 'Variables.xlsx';
%filename = '1.ELSA_Wave 7_v1.xlsx',
[ds,txt,raw] = xlsread(filename);

%% Distinguish between input and output
target=ds(:,2);
input=ds(:,3:43);

%% ***Data reduction of input variables with Self Organising Map (SOM):

% The SOM algorithm will classify the input vector, maintaining its
% distribution and topology

% transpose the input space (to 41 x 7573)
x_som = input';

% Specify Self-Organizing Map (2D:= 3 x 3)
LP.init_neighborhood = 5 % set initial neighbourhood size to 5
LP.steps = 500 % set ordering steps to 500
dimension1 = 5;
dimension2 = 4;
pos = randtop([dimension1 dimension2]); %specify random topology for original neuron locations
plotsom(pos);
d1 = dist(pos); % calculate Eucledian distances for topology of original neurons
som_net = selforgmap([dimension1 dimension2]);
% Choose Plot Functions
som_net.plotFcns = {'plotsomtop','plotsomnc','plotsomnd', ...
    'plotsomplanes', 'plotsomhits', 'plotsompos'};
som_net = configure(som_net, x_som)

%% Train the SOM
som_net.trainParam.epochs = 1000 % specify epochs
[som_net,tr_som] = train(som_net,x_som);
y_som = som_net(x_som);

%% View SOM and plots
view(som_net)
figure, plotsomtop(som_net)
figure, plotsomnc(som_net)
figure, plotsomnd(som_net)
figure, plotsomplanes(som_net)
figure, plotsomhits(som_net,x_som)
figure, plotsompos(som_net,x_som)

%% ***Backpropagation Neural Network (with raw data as inputs)

% transpose the input space (to 41 x 7573)
x_bp = input';
% transpose the output space (1 x 7573)
t_bp = target';

%Training
% For a list of all training functions type: help nntrain
%trainFcn = 'traingdm'; %Gradient descent with momentum backpropagation
trainFcn = 'traingdx';  %Gradient descent with momentum and adaptive learning rate backpropagation

% Create a Fitting Network
hiddenLayerSize = [50,30,20];
net_bp = fitnet(hiddenLayerSize,trainFcn);
net_bp.layers{1}.transferFcn = 'softmax'
net_bp.layers{2}.transferFcn = 'softmax' 
net_bp.layers{3}.transferFcn = 'logsig' 
net_bp.layers{3}.transferFcn = 'logsig' 
net_bp.outputs{1}.processFcns = 'logsig'


% For a list of all processing functions type: help nnprocess
net_bp.input.processFcns = {'removeconstantrows','mapminmax'};
net_bp.output.processFcns = {'removeconstantrows','mapminmax'};

% For a list of all data division functions type: help nndivide
net_bp.divideFcn = 'dividerand';  % Divide data randomly
net_bp.divideMode = 'sample';  % Divide up every sample
net_bp.divideParam.trainRatio = 60/100;
net_bp.divideParam.valRatio = 10/100;
net_bp.divideParam.testRatio = 30/100;

%Set parameters for training
net_bp.trainParam.min_grad=1e-10;
net_bp.trainParam.epochs=1000;
net_bp.trainParam.max_fail = 20;
net_bp.trainParam.lr=0.06
net_bp.trainParam.mc=0.01


% For a list of all performance functions type: help nnperformance
net_bp.performFcn = 'mse';  % Mean Squared Error

% For a list of all plot functions type: help nnplot
net_bp.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
[net_bp,tr_bp] = train(net_bp,x_bp,t_bp);

% Test the Network
y_bp = net_bp(x_bp);
e_bp = gsubtract(t_bp,y_bp);
performance = perform(net_bp,t_bp,y_bp)

% Recalculate Training, Validation and Test Performance
trainTargets = t_bp .* tr_bp.trainMask{1};
valTargets = t_bp .* tr_bp.valMask{1};
testTargets = t_bp .* tr_bp.testMask{1};
trainPerformance = perform(net_bp,trainTargets,y_bp)
valPerformance = perform(net_bp,valTargets,y_bp)
testPerformance = perform(net_bp,testTargets,y_bp)

% View the Network
view(net_bp)
figure, plotperform(tr_bp)
figure, plottrainstate(tr_bp)
figure, ploterrhist(e_bp)

%% Backpropagation Neural Net with SOM-derived inputs (***based on Dimitris' code***):

% define input space (already transposed; 20 x 7573)
x_bp_som = y_som;
% transpose the output space (1 x 7573)
t_bp_som = target';

%Training
% For a list of all training functions type: help nntrain
%trainFcn = 'traingdm'; %Gradient descent with momentum backpropagation
trainFcn = 'traingdx';  %Gradient descent with momentum and adaptive learning rate backpropagation

% Create a Fitting Network
hiddenLayerSize = [50,30,20];
net_bp_som = fitnet(hiddenLayerSize,trainFcn);
net_bp_som.layers{1}.transferFcn = 'softmax'
net_bp_som.layers{2}.transferFcn = 'softmax' 
net_bp_som.layers{3}.transferFcn = 'logsig' 
net_bp_som.outputs{1}.processFcns = 'logsig'


% For a list of all processing functions type: help nnprocess
net_bp_som.input.processFcns = {'removeconstantrows','mapminmax'};
net_bp_som.output.processFcns = {'removeconstantrows','mapminmax'};

% For a list of all data division functions type: help nndivide
net_bp_som.divideFcn = 'dividerand';  % Divide data randomly
net_bp_som.divideMode = 'sample';  % Divide up every sample
net_bp_som.divideParam.trainRatio = 60/100;
net_bp_som.divideParam.valRatio = 10/100;
net_bp_som.divideParam.testRatio = 30/100;

%Set parameters for training
net_bp_som.trainParam.min_grad=1e-5;
net_bp_som.trainParam.epochs=1000;
net_bp_som.trainParam.max_fail = 20;
net_bp_som.trainParam.lr=0.01
net_bp_som.trainParam.mc=0.7

% For a list of all performance functions type: help nnperformance
net_bp_som.performFcn = 'mse';  % Mean Squared Error

% For a list of all plot functions type: help nnplot
net_bp_som.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
[net_bp_som,tr_bp_som] = train(net_bp_som,x_bp_som,t_bp_som);

% Test the Network
y_bp_som = net_bp_som(x_bp_som);
e_bp_som = gsubtract(t_bp_som,y_bp_som);
performance = perform(net_bp_som,t_bp_som,y_bp_som)

% Recalculate Training, Validation and Test Performance
trainTargets = t_bp_som .* tr_bp_som.trainMask{1};
valTargets = t_bp_som .* tr_bp_som.valMask{1};
testTargets = t_bp_som .* tr_bp_som.testMask{1};
trainPerformance = perform(net_bp_som,trainTargets,y_bp_som)
valPerformance = perform(net_bp_som,valTargets,y_bp_som)
testPerformance = perform(net_bp_som,testTargets,y_bp_som)

% View the Network
view(net_bp_som)
figure, plotperform(tr_bp_som)
figure, plottrainstate(tr_bp_som)
figure, ploterrhist(e_bp_som)

%% Support Vector Machines
% Create a matrix with the training data in randomised order (remove first
% columns := id)
ds_svm=ds(:,2:43);
ds_svm_rand = ds_svm(randperm(size(ds_svm,1)),:);

%% Split into training and testing dataset (randomly 70%-train; 30% test)
numInst = size(ds_svm_rand,1);
idx = randperm(numInst);
numTrain = round(numInst*(70/100)); 
numTest = numInst - numTrain;

svm_train = ds_svm_rand(idx(1:numTrain),:);  
svm_test = ds_svm_rand(idx(numTrain+1:end),:);

%% Train, validate, and test SVM with linear kernel
[svm_kernel_linear_trained, validationAccuracy_svm_kernel_linear] = svm_kernel_linear(svm_train)
yfit_svm_kernel_linear = svm_kernel_linear_trained.predictFcn(svm_test(:,2:42));
confusion_svm_kernel_linear = confusionmat(svm_test(:,1),yfit_svm_kernel_linear)
stats_svm_kernel_linear = confusionmatStats(svm_test(:,1),yfit_svm_kernel_linear)

%% Train, validate, and test SVM with quadratic kernel
[svm_kernel_quad_trained, validationAccuracy_svm_kernel_quad] = svm_kernel_quad(svm_train);
yfit_svm_kernel_quad = svm_kernel_quad_trained.predictFcn(svm_test(:,2:42));
confusion_svm_kernel_quad = confusionmat(svm_test(:,1),yfit_svm_kernel_quad)
stats_svm_kernel_quad = confusionmatStats(svm_test(:,1),yfit_svm_kernel_quad)

%% Train, validate, and test SVM with cubic kernel 
[svm_kernel_cubic_trained, validationAccuracy_svm_kernel_cubic] = svm_kernel_cubic(svm_train);
yfit_svm_kernel_cubic = svm_kernel_cubic_trained.predictFcn(svm_test(:,2:42));
confusion_svm_kernel_cubic = confusionmat(svm_test(:,1),yfit_svm_kernel_cubic)
stats_svm_kernel_cubic = confusionmatStats(svm_test(:,1),yfit_svm_kernel_cubic)

%% Train, validate, and test SVM with fine gaussian kernel 
[svm_kernel_fgauss_trained, validationAccuracy_svm_kernel_fgauss] = svm_kernel_fgauss(svm_train);
yfit_svm_kernel_fgauss = svm_kernel_fgauss_trained.predictFcn(svm_test(:,2:42));
confusion_svm_kernel_fgauss = confusionmat(svm_test(:,1),yfit_svm_kernel_fgauss)
stats_svm_kernel_fgauss = confusionmatStats(svm_test(:,1),yfit_svm_kernel_fgauss)

%% Train, validate, and test SVM medium gaussian kernel 
[svm_kernel_mgauss_trained, validationAccuracy_svm_kernel_mgauss] = svm_kernel_mgauss(svm_train);
yfit_svm_kernel_mgauss = svm_kernel_mgauss_trained.predictFcn(svm_test(:,2:42));
confusion_svm_kernel_mgauss = confusionmat(svm_test(:,1),yfit_svm_kernel_mgauss)
stats_svm_kernel_mgauss = confusionmatStats(svm_test(:,1),yfit_svm_kernel_mgauss)

%% Train, validate, and test SVM with coarse gaussian kernel 
[svm_kernel_cgauss_trained, validationAccuracy_svm_kernel_cgauss] = svm_kernel_cgauss(svm_train);
yfit_svm_kernel_cgauss = svm_kernel_cgauss_trained.predictFcn(svm_test(:,2:42));
confusion_svm_kernel_cgauss = confusionmat(svm_test(:,1),yfit_svm_kernel_cgauss)
stats_svm_kernel_cgauss = confusionmatStats(svm_test(:,1),yfit_svm_kernel_cgauss)

