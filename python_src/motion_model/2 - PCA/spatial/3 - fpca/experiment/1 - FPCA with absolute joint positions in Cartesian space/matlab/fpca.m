clc
clear all
%% load the data
load walk_left_featureVector
% the loaded data is a 4d array: n_samples * n_frames * n_joints * (x, y ,z)
% position
%% compute mean motion (numeric approach)
[n_samples n_frames n_joints dim] = size(data);
sum = zeros(1, n_frames, n_joints, dim);
for i = 1: n_samples;
    sum = sum + data(i, :, :, :);
end
mean_motion = sum./n_samples;
%% centralize data
centered_data = zeros(n_samples, n_frames, n_joints, dim);
for i = 1:n_samples;
    centered_data(i, :, :, :) = data(i,:,:,:) - mean_motion;
end
%% use cubic spline to fit data
splines = [];
for i = 1: n_samples;
    tmp = zeros(n_joints*dim, n_frames); % matrix to store one motion segment data
    for j = 1: n_frames;
        oneframe_data = centered_data(i,j,:,:);
        oneframe_data = reshape(oneframe_data, [n_joints, dim]);
        oneframe_data1D = reshape(oneframe_data', [n_joints * dim, 1]);
        tmp(:, j) = oneframe_data1D;
    end
    sp = csapi(1:n_frames, tmp);
    splines = [splines ; sp];
end
%% compute covariance matrix
covmat = zeros(n_samples, n_samples);
for i = 1: n_samples;
    for j = 1: n_samples;
        covmat(i,j) = int_spline(splines(i), splines(j));
    end
end
%% eigenvalue decomposition
[V, D] = eig(covmat);
eigenvalues = diag(D);
[eigenvalues, idx] = sort(eigenvalues, 'descend');
eigenvectors = D(:, idx);
cumvariance = cumsum(eigenvalues);
cumvariance = cumvariance./cumvariance(end);
npc = find(cumvariance >= 0.99, 1);
eigenvalues = eigenvalues(1:npc);
eigenvectors = eigenvectors(:, 1:npc);
