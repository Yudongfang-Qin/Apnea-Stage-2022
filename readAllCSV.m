function [allFileInfo,data] = readAllCSV(filePath)

filePattern = sprintf('%s/*.csv', filePath); 
allFileInfo = dir(filePattern);

for i = 1 : round(length(allFileInfo))
    fname = allFileInfo(i).folder + "/" + allFileInfo(i).name;
    
    try
        curData = csvread(fname, 1);
        data(i).data = curData;
    catch
        fprintf(fname);
    end


end


end


