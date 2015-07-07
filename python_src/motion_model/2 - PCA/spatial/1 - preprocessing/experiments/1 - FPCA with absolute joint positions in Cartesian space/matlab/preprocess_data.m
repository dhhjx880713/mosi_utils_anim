clc;
clear all;
%% load json file
filename = 'walk_leftStance_featureVector.json';
raw_data = loadjson(filename);
filenames = fieldnames(raw_data); % read fieldnames
%% 
numberOfJoints = 19;
numberOfFrames = 47;
numberOfSamples = length(filenames);
data = zeros(numberOfSamples, numberOfFrames, numberOfJoints, 3);
%% convert raw data to a matrix
filenames = char(filenames);
for i = 1: numberOfSamples;
    % get the data from data struct
    pointarray_onesample = getfield(raw_data, filenames(i, :));
    % reshape point array of one sample as a 3d matrix: frame_number * 
    % joint_number * 3
    tmp = zeros(numberOfFrames, numberOfJoints, 3);
    for j = 1: numberOfFrames;
        for k = 1: numberOfJoints;
            tmp(j, k, :) = pointarray_onesample((j-1)*numberOfJoints + k, :);
        end
    end
    data(i, :, :, :) = tmp;
end
%% store data as mat file
save('walk_left_featureVector.mat', 'data')
