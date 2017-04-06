function [target,input]=read_file(filename)
    %% read filename with xlsread
    [ds,txt,raw] = xlsread(filename);
    %% distinguish between input and output
    target=ds(:,2);
    input=ds(:,3:43);
end