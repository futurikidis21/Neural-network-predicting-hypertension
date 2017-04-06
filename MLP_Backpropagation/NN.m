function [net_bp,y_bp,tr_bp,x_bp,e_bp,performance,prediction_pcr,accuracy_test,accuracy_overall,ttime]=NN(input,target,hiddenLayerSize,trainf,trans,outk,lay,learning_rate,momentum)
% Backpropagation Neural Net (***Dimitris' code***):

% transpose the input space (to 41 x 7573) according to user input for
% either SOM or raw data-testing the data structure fed by user input
if size(input)>20
    x_bp = input';
    t_bp = target';
else
    x_bp = input;
    t_bp = target';
end

%training function set
trainFcn = trainf;  %neural network training function

%selection of layer size given as user input
if lay==1
    NN_size=[hiddenLayerSize];%hidden layer size
    net_bp = fitnet(NN_size,trainFcn); % fit network
    net_bp.layers{1}.transferFcn = trans; %hidden layer activation function
    net_bp.layers{2}.transferFcn = outk;  %output layer activation function
else 
    NN_size=[hiddenLayerSize,round(0.8*hiddenLayerSize)]; %two layer NN size
    net_bp = fitnet(NN_size,trainFcn);
    net_bp.layers{1}.transferFcn = trans;
    net_bp.layers{2}.transferFcn = trans; 
    net_bp.layers{3}.transferFcn = outk;
end

%configuration of NN and data preperation
net_bp.input.processFcns = {'removeconstantrows','mapminmax'};
net_bp.output.processFcns = {'removeconstantrows','mapminmax'};

net_bp.divideFcn = 'dividerand';  % Divide data randomly
net_bp.divideMode = 'sample';  % Divide up every sample
net_bp.divideParam.trainRatio = 60/100;
net_bp.divideParam.valRatio = 10/100;
net_bp.divideParam.testRatio = 30/100;

%set configuration parameters for training and early stoping
net_bp.trainParam.min_grad=1e-10;
net_bp.trainParam.epochs=1000;
net_bp.trainParam.max_fail = 20;
net_bp.trainParam.lr=learning_rate;
net_bp.trainParam.mc=momentum;

%GUI and parameters print control
net_bp.trainParam.showWindow=1;
net_bp.trainParam.show = NaN;

% system performance function
net_bp.performFcn = 'mse';  % Mean Squared Error

%net_bp.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...'plotregression', 'plotfit'};

% train the network with the configuration and data given
[net_bp,tr_bp] = train(net_bp,x_bp,t_bp);

% Test the Network
y_bp = net_bp(x_bp);
e_bp = gsubtract(t_bp,y_bp);
performance = perform(net_bp,t_bp,y_bp);

%Calculate test performance with a thresshold of 0.5 for 0-1 output
predicted_treshold=round(y_bp); %round output to 0-1
test_targets=t_bp.*tr_bp.testMask{1}; %apply mask for test to filet for test values only for targets
test_result=predicted_treshold.*tr_bp.testMask{1}; %apply mask for test to filet for test values only for output
compare=test_result'==test_targets'; %compare test output and target and return 0-1 array
accuracy_test=sum(compare,1)/2272; %calculate test accuracy for 30% of the data (test data)
prediction_pcr = sum(y_bp'>0.5)/size(y_bp',1); %calculate % of hypertension prediction of total output

%Calculate overall performance with a thresshold of 0.5 for 0-1 output
compare_overall=predicted_treshold'==t_bp'; %compare overall ouput to target output
accuracy_overall=sum(compare_overall,1)/7573; %calculate accuracy of overall output
% fprintf('Model overall accuracy %2f\n',accuracy_overall)
% fprintf('Model testing accuracy %2f\n',accuracy_test)
ttime=tr_bp.time(:,tr_bp.num_epochs+1); %return last epoch timing for comparison of neural nets configuration

% % View the Network
% view(net_bp)
% figure, plotperform(tr_bp);
% figure, plottrainstate(tr_bp);
% figure, ploterrhist(e_bp);
end