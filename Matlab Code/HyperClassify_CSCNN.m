
clc;
clear all;
%% load the data_set , the data_set must be .tif and have .hdr file.
%% when your net was trained and you want to reuse one, your new data_set must have the same spectral band
%% and for crate te patch mast use the faunction crateImagePatche.
info = enviinfo('moqan.tif');
hcube = hypercube('moqan.tif',info.Wavelength);

%% show the false_color version of data_set
rgbImg = colorize(hcube,'method','rgb');
imshow(rgbImg)

%% load the train data
gt = imread('Clip_GroundTruth.tif');
save('myTiff2mat','gt');
gtLabel = load('gt.mat');
gtLabel = gtLabel.gt;
numClasses = 7;

%% preprocess training data
%% when our data is hyper 
dimReduction = 2;
imageData = hyperpca(hcube,dimReduction);
sd = std(imageData,[],3);
imageData = imageData./sd; 
windowSize = 15;
inputSize = [windowSize windowSize dimReduction];
[allPatches,allLabels] = createImagePatchesFromHypercube(imageData,gtLabel,windowSize);
moqannDataTransposed = permute(allPatches,[2 3 4 1]);
dsAllPatches = augmentedImageDatastore(inputSize,moqannDataTransposed,allLabels);
patchesLabeled = allPatches(allLabels>0,:,:,:);
patchLabels = allLabels(allLabels>0);
numCubes = size(patchesLabeled,1);
patchLabels = categorical(patchLabels);
[trainingIndex,validationIndex,testInd] = dividerand(numCubes,0.3,0.7,0);
dataInputTrain = patchesLabeled(trainingIndex,:,:,:);
dataLabelTrain = patchLabels(trainingIndex,1);
dataInputTransposeTrain = permute(dataInputTrain,[2 3 4 1]); 
dataInputVal = patchesLabeled(validationIndex,:,:,:);
dataLabelVal = patchLabels(validationIndex,1);
dataInputTransposeVal = permute(dataInputVal,[2 3 4 1]);
imdsTrain = augmentedImageDatastore(inputSize,dataInputTransposeTrain,dataLabelTrain);
imdsTest = augmentedImageDatastore(inputSize,dataInputTransposeVal,dataLabelVal);

%% Create CSCNN Classification Network
layers = [
    image3dInputLayer(inputSize,"Name","Input","Normalization","none")
    convolution3dLayer([3 3 7],8,"Name","conv3d_1","Padding",[1 0 0;1 0 1],"Stride",[2 2 2])
    reluLayer("Name","Relu_1")
    convolution3dLayer([3 3 5],16,"Name","conv3d_2","Padding","same","Stride",[2 2 2])
    reluLayer("Name","Relu_2")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3","Padding","same","Stride",[2 2 2])
    reluLayer("Name","Relu_3")
    convolution3dLayer([3 3 1],8,"Name","conv3d_4","Padding","same","Stride",[2 2 2])
    reluLayer("Name","Relu_4")
    fullyConnectedLayer(256,"Name","fc1")
    reluLayer("Name","Relu_5")
    dropoutLayer(0.4,"Name","drop_1")
    fullyConnectedLayer(128,"Name","fc2")
    dropoutLayer(0.4,"Name","drop_2")
    fullyConnectedLayer(7,"Name","fc3")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","output")];
lgraph = layerGraph(layers);

%% Visualize the network using Deep Network Designer.
numEpochs = 100;
miniBatchSize = 256;
initLearningRate = 0.001;
momentum = 0.9;
learningRateFactor = 0.01;

options = trainingOptions('adam', ...
    'InitialLearnRate',initLearningRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'LearnRateDropFactor',learningRateFactor, ...
    'MaxEpochs',numEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThresholdMethod','l2norm', ...
    'GradientThreshold',0.01, ...
    'VerboseFrequency',100, ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',100);

%% Train the Network
 net = trainNetwork(imdsTrain,lgraph,options);

%% Classify Hyperspectral Image Using Trained CSCNN
predictionTest = classify(net,imdsTest);
accuracy = sum(predictionTest == dataLabelVal)/numel(dataLabelVal);
disp(['Accuracy of the test data = ', num2str(accuracy)])

%% 
prediction = classify(net,dsAllPatches);
prediction = double(prediction);

%%
patchesUnlabeled = find(allLabels==0);
prediction(patchesUnlabeled) = 0;

%%
[m,n,d] = size(imageData);
moqannPrediction = reshape(prediction,[n m]);
moqannPrediction = moqannPrediction';

%% Display the ground truth and predicted classification
cmap = parula(numClasses);
figure
tiledlayout(1,2,"TileSpacing","Tight")
nexttile
imshow(gtLabel,cmap)
title("Ground Truth Classification")

nexttile
imshow(moqannPrediction,cmap)
colorbar
title("Predicted Classification")
%% save pretrained net
AminiCSCNNnet = net;
 save AminiCSCNNnet
