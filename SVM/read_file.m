function [target,input]=read_file(filename)
    %read filename with xlsread
    [ds,txt,raw] = xlsread(filename);
    %% Distinguish between input and output
    target=ds(:,2);
    input=ds(:,3:43);
end