% Based on James Hays, Brown University

%This function will sample SIFT descriptors from the training images,
%cluster them with kmeans, and then return the cluster centers.

function vocab = build_vocabulary( image_paths, vocab_size )

for x = 1:length(image_paths) %loop through image paths
    img = imread(image_paths{x,1}); %read in current image
    grayImage = single(rgb2gray(img)); %convert image to grayscale

    step = 8; %can be changed if needed
    [~, SIFT_features] =  vl_dsift(grayImage, 'step', step,'fast'); %obtains SIFT features
    
    %Store and concatonate features
    feature = single([]);
    feature = single(cat(2,feature, SIFT_features));      
end
    %Get centroid for each word
    [centers, ~] = vl_kmeans(feature, vocab_size);
    vocab = centers;
    
end
