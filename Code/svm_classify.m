% Based on James Hays, Brown University

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters.

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)


imgLabels = zeros(1500, 15); %initialising matrix for binary representation of image labels
s = zeros(15,1500); %initialising score matrix
predicted_categories = cell(1500,1); %initialising output matrix

start = 1;
finish = 100;

%Loops through each category 
for j = 1: 15
    for i = 1: 1500
        if(i >= start && i <= finish)
            imgLabels(i,j) = 1;
        else
            imgLabels(i,j) = -1;
        end
    end
    start = start + 100;
    finish = finish + 100;
end

for i = 1: 15
    [W(:,i),B(:,i)] = vl_svmtrain(train_image_feats', imgLabels(:,i), 0.0000001); %trains the SVM from the features and labels.
end

for i = 1: 1500
    for p = 1: 15
        s(p,i) = W(:,p)' * test_image_feats(i,:)' + B(:,p); %counts the scores of each category
    end
end

[~,index] = max(s,[],1); %determines and stores max distance for each image with the relevant index

categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
       'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
       'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};

for i = 1: 1500
    predicted_categories(i,1) = categories(index(1,i)); %Stores index of maximum score for each image
end

end