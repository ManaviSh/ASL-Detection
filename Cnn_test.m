clear all
clc

imds = imageDatastore('MerchData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
imds_new = shuffle(imds);
 figure;
perm = randperm(50,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds_new.Files{perm(i)});
end
labelCount = countEachLabel(imds_new)
img = readimage(imds_new,1);
size(img)
[imds_newTrain,imds_newValidation] = splitEachLabel(imds_new,0.3,'randomize');
layers = [
    imageInputLayer([200 200 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(29)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imds_newValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(imds_newTrain,layers,options);
[YPred,scores] = classify(net,imds_newValidation);
idx = randperm(numel(imds_newValidation.Files),2);
figure
for i = 1:2
    subplot(2,1,i)
    I = readimage(imds_newValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end
% YValidation = im1.Labels;
% accuracy = sum(YPred == YValidation)/numel(YValidation)
% clear camera;
% camera = webcam();
% while true
% 
% I = camera.snapshot;
% sz = net.Layers(1).InputSize
% I = I(1:sz(1),1:sz(2),1:sz(3));
% label = classify(net,I);
% image(I)
% title(char(label));
%  drawnow;
% end