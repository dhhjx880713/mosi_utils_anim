library('fda')
nharm <- 4
root <- readRDS('data/walk_leftStance_fd.RData')
pcaobj <- pca.fd(root, nharm=nharm)
pcavarmax <- varmx.pca.fd(pcaobj)
scores <- pcavarmax$scores
n_samples <- dim(root$coefs)[2]


# stack all scores to on list, a.k.a. each file has a 1D-list with [num_harmonics*num_scores] elements
stacked_scores_list <- list()

for (i in 1:n_samples){
    stacked_scores_list[i] = list(c(scores[i,,]))
}
stacked_scores <- matrix(unlist(stacked_scores), nrow=dim(scores)[1])

# perform a standard PCA on this new dataset
# tol == 0 means, we want all eigen vectors.
pca <- prcomp(stacked_scores, tol=0)
pca_dim <- dim(stacked_scores)[2]

# calculate the eigenvalues and the cumsum
eigenvalues = pca$sdev*pca$sdev
rel_eigenvalues = eigenvalues / sum(eigenvalues)
rel_cumsum = cumsum(rel_eigenvalues)

# get the number of components 
# NOTE: The number is still very high
fraction = 1
npc = which(rel_cumsum >= fraction)[1]

# take the first [npc] components
eigenvectors <- pca$rotation[1:npc,]
stacked_scores_centered <- stacked_scores
for (i in 1:pca_dim){
    stacked_scores_centered[,i] <- stacked_scores[,i] - pca$center[i]
}

lowdim <- list()
for (i in 1:n_samples){
    lowdim[i] <- list(eigenvectors %*% stacked_scores_centered[i,])
}
lowdim <- matrix(unlist(lowdim), nrow=n_samples)


# backproj temp
samplescore_center <- t(eigenvectors) %*% lowdim[1,]
samplescore <- samplescore_center + pca$center


test <- t(eigenvectors) %*% pca$x[1,]


#######################
# Test backprojection #
#######################

# set the file to be tested
sampleid <- 100


# do the backprojection
#    1. backtransform the standard pca
samplescore_center <- lowdim[sampleid,] %*% t(eigenvectors)
samplescore <- samplescore_center + pca$center

diff = stacked_scores[sampleid,] - samplescore

#    2. backtransform fpca
# create a matrix with the right size by coping one.
# initialize a few variables
harmonic_coefs = coef(pcavarmax$harmonics)
mean_coefs = coef(pcavarmax$meanfd)
nbasis <- dim(harmonic_coefs)[1]
new_coefs <- root$coefs[,1,]
ndim <- dim(harmonic_coefs)[3]
for (i in 1:nbasis){

    for (dim in 1:ndim){
        scores_i = (harmonic_coefs[i,,dim] * samplescore[,dim]) 
        new_coefs[i,dim] = sum(scores_i) + mean_coefs[i,1,dim]
    }
}

# create the new fd object
newfd <- fd(new_coefs, pcavarmax$harmonics$basis)
# create an fd object based on the original coefs
oldfd <- fd(root$coefs[,sampleid,], pcavarmax$harmonics$basis)

# Evaluate the two functions at a set of points
t = seq(0,46)

frames = eval.fd(t, newfd)
frames_old = eval.fd(t, oldfd)

# save the frames
saveRDS(frames, 'new_frames.RData')
saveRDS(frames_old, 'original_frames.RData')

# get stacked coefs for normal pca
#coef_stacked <- NULL
#for (i in 1:n_samples){
#    coef_stacked <- cbind(coef_stacked, c(root$coefs[,i,]))   
#}
#coef_stacked <- t(coef_stacked)
#
#pca <- prcomp(coef_stacked, tol=0)
#eigenvectors <- pca$rotation[,1:4]
#lowdim <- coef_stacked %*% eigenvectors
#
## test:
#sampleid <- 100
#backprojected <-  lowdim[sampleid,] %*% t(eigenvectors)
#
#new_scores = scores[1,,]
##new_scores[,1] = backprojected[
##
##print(dim(backprojected))