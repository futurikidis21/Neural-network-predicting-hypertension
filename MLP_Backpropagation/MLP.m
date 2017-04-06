%% High blood pressure and socio-demographic profiling:       %
% A comparative analysis of Back-propagation Neural Networks  %
% versus Support Vector Machines in predicting                %
% the occurance of high blood-pressure based on responses to  %
% the English Longitudinal Study of Ageing                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 1: Read file to examing
%Set path to file
    path='/Users/squirel/Library/Mobile Documents/com~apple~CloudDocs/NN_Coursework/NN-Matlab/';
    %Set filename of excel file
    filename='Variables.xlsx';
    %Call function to read file and seperate to targets and inputs
    [target,input]=read_file(strcat(path,filename));
    
%% Step 2:Training of self organising map network to obtain clustered comparison outputs
    y_som=SOM_Function(input); %Call SOM function
    
%% Step 3: Set configuration parameters with training and activation functions along with parameters to be tested

    hiddenLayerSize=50;%initial hidden layer size
    outfcn={'tansig';'logsig';};%output layer function
    trainf={'traingdx';'trainrp';};%training functions to be tested
    trans={'logsig';'tansig';'purelin';};%activation functions to be tested
    lay=1%layer size selection choises of 1-2
    learning_rate=0.01;%initial learning rate
    momentum=0.1;%initial momentum
    log_raw=[];%array to store configuration and results from raw training
    total_raw=[];%array to store results from raw training
    log_som=[];%array to store configuration and results from raw training
    total_som=[];%array to store configuration and results from raw training
    lay=2;%placeholder for layers
    
%% Step 4: Loop through the configuration set on step 3 and train and test raw input data and som input data from step 2
% The data are stored in cell arrays along with the network configuration
% for further analysis

%Loop through architectures and store accuracy and configuration results
    for k=1:size(outfcn,1) %loop on the activation function
        for i=1:size(trainf,1) %loop through training functions
            for j=1:size(trans,1) %loop through activation functions
                while learning_rate <=0.09 %learning maximum thresshold
                    while momentum<1 %momentum rate maximum thresshold
                        while hiddenLayerSize<=300 %layer size thresshold set to 300 and loop to increase
                            
                            traini=trainf{i}; %set train parameter
                            transj=trans{j}; %set activation parameter
                            outk=outfcn{k}; % set ouput parameter
                            
                            %Train on raw inputs
                            fprintf('Trainining with raw input on configuration: \n Layer Size: %d ,Training Function: %s ,Transfer Function: %s ,Layers: %i ,Learning Rate: %f ,Momentum: %f \n\n',hiddenLayerSize,traini,transj,lay,learning_rate,momentum)
                            [net_bp,y_bp,tr_bp,x_bp,e_bp,performance,prediction_pcr,accuracy_test,accuracy_overall]=NN(input,target,hiddenLayerSize,traini,transj,outk,lay,learning_rate,momentum);
                            fprintf('Model accuracy for test set %2.2f\n\n',accuracy_test*100)
                            
                            %Save raw train configuration and results
                            to_add={'Raw_Data',hiddenLayerSize,traini,transj,outk,lay,learning_rate,momentum,prediction_pcr,accuracy_test,accuracy_overall,tr_bp,e_bp,x_bp,y_bp,tr_bp.num_epochs,ttime};
                            log_raw=[log_raw;to_add]; % save on array log file and results
                            total_raw=[total_raw;y_bp]; % save raw output on array
                            
                            %Train on SOM inputs
                            fprintf('Trainining with SOM input on configuration: \n Layer Size: %d ,Training Function: %s ,Transfer Function: %s ,Layers: %i ,Learning Rate: %f ,Momentum: %f \n\n',hiddenLayerSize,traini,transj,lay,learning_rate,momentum)
                            [net_bp,y_bp,tr_bp,x_bp,e_bp,performance,prediction_pcr,accuracy_test,accuracy_overall]=NN(y_som,target,hiddenLayerSize,traini,transj,outk,lay,learning_rate,momentum);
                            fprintf('Model accuracy for test set %2.2f\n\n',accuracy_test*100)
                            
                            %Save SOM train configuration and results
                            to_add={'SOM_Data',hiddenLayerSize,traini,transj,outk,lay,learning_rate,momentum,prediction_pcr,accuracy_test,accuracy_overall,tr_bp,e_bp,x_bp,y_bp,tr_bp.num_epochs,ttime};
                            log_som=[log_som;to_add];
                            total_som=[total_som;y_bp];
                            
                            %increase layer size
                            hiddenLayerSize=hiddenLayerSize+50;
                        end
                        momentum=momentum+0.1; %increase momentum
                        hiddenLayerSize=50; %reset layer size
                    end
                    learning_rate=learning_rate+0.01; %increase learning rate
                    hiddenLayerSize=50; %reset layer size
                    momentum=0.01; %reset momentum
                end
                learning_rate=0.01; %reset learning rate
                momentum=0.01; %reset momentum
                hiddenLayerSize=50; %reset layer size
            end
            learning_rate=0.01; %reset learning rate
            momentum=0.01; %reset momentum
            hiddenLayerSize=50; %reset layer size
        end
        learning_rate=0.01; %reset learning rate
        momentum=0.01; %reset momentum
        hiddenLayerSize=50; %reset layer size
    end
%% Step 5: Save workspace from step 4 to file NN_Output.mat
    filename='NN_Output.mat'
    save(filename,'log_raw','total_raw','log_som','total_som') %Save workspace data
%% Step 6: Testing optimal settings to repeat train neural net from previous step
%Re-run step 1 to initialise the raw inputs

%Subsection a: Optimum settings selected
    trainf=['traingdx']; %Training function set
    trans=['tansig']; %Hidden layer function set
    outk=['tansig']; %Output layer function set
    lay=1; %Number of layers function set
    learning_rate=0.04; %Learning rate set
    momentum=0.11; %Momentum rate set
    hiddenLayerSize=150; %Hidden layer size
    retrain_log=[];
    retrain_results=[];
    
%Subsection b: Run the 10 instances of the neural network and store performance
for i=1:10
    %Neural net function configuration set
    [net_bp,y_bp,tr_bp,x_bp,e_bp,performance,prediction_pcr,accuracy_test,accuracy_overall,ttime]=NN(input,target,hiddenLayerSize,trainf,trans,outk,lay,learning_rate,momentum);
    fprintf('Model accuracy for test set %2.2f\n',accuracy_test*100)
    fprintf('Model overall accuracy %2.2f\n',accuracy_overall*100)
    %Log results to arrays
    to_add={hiddenLayerSize,trainf,trans,outk,lay,learning_rate,momentum,prediction_pcr,accuracy_test,accuracy_overall,tr_bp,e_bp,x_bp,y_bp,tr_bp.num_epochs,ttime};
    retrain_log=[retrain_log;i,to_add]; %Store log of neural net along with metrics
    retrain_results=[retrain_results;y_bp]; %Store outputs
end
%% Step 7: Save workspace results arrays to file NN_Retrain.mat
    filename='NN_Retrain.mat'
    save(filename,'retrain_log','retrain_results')



