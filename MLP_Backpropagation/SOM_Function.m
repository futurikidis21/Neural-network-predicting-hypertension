function [y_som,som_net,tr_som]=som(input)
    %Data reduction of input variables with Self Organising Map (SOM)
    % The SOM algorithm will classify the input vector, maintaining its
    % distribution and topology

    % transpose the input space (to 41 x 7573)

    % specify Self-Organizing Map (2D:= 3 x 3)
    LP.init_neighborhood = 5; % set initial neighbourhood size to 5
    LP.steps = 500; % set ordering steps to 500
    dimension1 = 5;
    dimension2 = 4;
    
    pos = randtop([dimension1 dimension2]); %specify random topology for original neuron locations
    plotsom(pos);
    d1 = dist(pos); % calculate Eucledian distances for topology of original neurons
    som_net = selforgmap([dimension1 dimension2]);

    % choose Plot Functions
    som_net.plotFcns = {'plotsomtop','plotsomnc','plotsomnd', ...
    'plotsomplanes', 'plotsomhits', 'plotsompos'};
    som_net = configure(som_net, input');
    
    %GUI and parameters print control
    som_net.trainParam.showWindow=0;
    som_net.trainParam.show = NaN;
    som_net.trainParam.showCommandLine = false; 
    
    %train the SOM
    som_net.trainParam.epochs = 1000; % specify epochs
    [som_net,tr_som] = train(som_net,input');
    y_som = som_net(input');
    
    %View SOM and plots
    %view(som_net)
    %figure, plotsomtop(som_net)
    %figure, plotsomnc(som_net)
    %figure, plotsomnd(som_net)
    %figure, plotsomplanes(som_net)
    %figure, plotsomhits(som_net,input')
    %figure, plotsompos(som_net,input')
end

