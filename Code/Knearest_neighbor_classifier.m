function [ output ] = Knearest_neighbor_classifier(train_image_feats,train_labels, test_image_feats, k)

%Using the pdist2 function, it compares the test data to the training data
%and returns a matrix which contains the distance between the points in
%acending order - storing the index for each point for later reference.
[~,i] = pdist2(test_image_feats, train_image_feats, 'euclidean', 'Smallest', k);

totalIMG = 1500; %Total number of images in the data sets
results = cell(k,totalIMG); %stores results 

for p = 1:totalIMG %loops through images
    for j = 1:k
        results(j,p) = train_labels(i(j,p)); %stores test labels
    end
end

output = cell(1,length(train_labels)); %creating cell array
results = results.'; %transpose results matrix


% Looping through the rows in results to find each set of unique labels for
% the current image + finding which label has occured most often for the
% current image. It is stored and used as the label for the image. 
for p = 1: length(train_labels) % Loops through labels
    y = results(p,:); % sets y value to current row of results
    max = 0; 

    for z = 1:length(y)
        n = length(find(strcmp(y{z}, y)));
        if(n > max)
            max = n;
            a = y(z);
        end
        
    end
    
    output(1,p) = a;
end

output = output.';
end


