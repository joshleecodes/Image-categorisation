% Implementated according to the starter code prepared by James Hays, Brown University
% Michal Mackiewicz, UEA, March 2015

function image_feats = get_bags_of_sifts(image_paths)

load('vocab.mat')
image_feats = zeros(1500,128); %allocating matrix for output

%Greyscale
%{
for x = 1:length(image_paths) %loop through image paths
    img = imread(image_paths{x,1}); %read in current image
    grayImage = single(rgb2gray(img)); %convert image to grayscale

    step = 8; %can be changed if needed
    [~, SIFT_features] =  vl_dsift(grayImage, 'step', step,'fast'); %obtains SIFT features
        
    D = vl_alldist2(single(SIFT_features),vocab); % Returns the pairwise distance of two columns 
    
    [~, minIndex] = min(D,[],2); %Find minimum distance
    clusterHisto = hist(minIndex,128); %Create histogram with minimum values
    clusterHisto = normr(clusterHisto); %Normalise
    image_feats(x,:) = clusterHisto; % Store histogram into feature vector
end 
%}



%Colour(RGB)
for x = 1:length(image_paths) %loop through image paths
    img = imread(image_paths{x,1}); %read in current image
    img = single(img);

    step = 8; %can be changed if needed
    [~, redVal] =  vl_dsift(img(:,:,1), 'step', step,'fast'); %obtains SIFT features of red channel
    [~, greenVal] =  vl_dsift(img(:,:,2), 'step', step,'fast'); %obtains SIFT features of green channel
    [~, blueVal] =  vl_dsift(img(:,:,3), 'step', step,'fast'); %obtains SIFT features of blue channel
        
    R = vl_alldist2(single(redVal),vocab); % Returns the pairwise distance of two columns
    G = vl_alldist2(single(greenVal),vocab); % Returns the pairwise distance of two columns
    B = vl_alldist2(single(blueVal),vocab); % Returns the pairwise distance of two columns

    RGB = cat(1,R,G,B); %Concatonates the pairwise distance for each channel. 
    
    m = min(RGB,[],2); %Find minimum distance
    clusterHisto = hist(m,128); %Create histogram with minimum values
    clusterHisto = normr(clusterHisto); %Normalise
    image_feats(x,:) = clusterHisto; % Store histogram into feature vector
end 

%hsv
%{
for x = 1:length(image_paths) %loop through image paths
    img = imread(image_paths{x,1}); %read in current image
    img = rgb2hsv(img);
    img = single(img);

    step = 8; %can be changed if needed
    [~, hVal] =  vl_dsift(img(:,:,1), 'step', step,'fast'); %obtains SIFT features of first channel
    [~, sVal] =  vl_dsift(img(:,:,2), 'step', step,'fast'); %obtains SIFT features of second channel
    [~, vVal] =  vl_dsift(img(:,:,3), 'step', step,'fast'); %obtains SIFT features of third channel
        
    H = vl_alldist2(single(hVal),vocab); % Returns the pairwise distance of two columns
    S = vl_alldist2(single(sVal),vocab); % Returns the pairwise distance of two columns
    V = vl_alldist2(single(vVal),vocab); % Returns the pairwise distance of two columns

    RGB = cat(1,H,S,V); %Concatonates the pairwise distance for each channel. 
    
    m = min(RGB,[],2); %Find minimum distance
    clusterHisto = hist(m,128); %Create histogram with minimum values
    clusterHisto = normr(clusterHisto); %Normalise
    image_feats(x,:) = clusterHisto; % Store histogram into feature vector
end 
%} 

end