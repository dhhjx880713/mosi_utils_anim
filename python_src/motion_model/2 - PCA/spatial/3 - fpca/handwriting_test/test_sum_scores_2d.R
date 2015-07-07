# code for the fpca based on the book Functional Data Analysis with Matlab and R 
# approximate the original data via a weighted linear combination of eigenfunctions
# with the weights given by the sum of the scores returned by the pca.fd function for each dimension
library(fda)

#load the data
load('handwrit.rda')
print(dim(handwrit))
numframes = 2300
#create smooth bivariate function representation for each replication
fdarange = c(0, numframes)
fdabasis = create.bspline.basis(fdarange, 105, 6)#create a b-spline basis system with 105 functions with degree 6 on equally spaced knots
fdatime = seq(0, numframes, len=1401)
fdafd = smooth.basis(fdatime, handwrit, fdabasis)$fd
fdafd$fdnames[[1]] = "Milliseconds"
fdafd$fdnames[[2]] = "Replications"
fdafd$fdnames[[3]] = list("X", "Y")

# set the number of eigenfunctions/harmonics to four and perform fPCA
nharm = 4
fdapcaList = pca.fd(fdafd, nharm)
#plot.pca.fd(fdapcaList)
fdarotpcaList = varmx.pca.fd(fdapcaList)
print(dim(fdarotpcaList$scores))


# extract the sample, mean and harmonic coefficients and stack them together
samplecoefs = fdafd$coefs
harmonics <-fdarotpcaList$harmonics
mean <-fdarotpcaList$mean
#b= coef(fdafd[3]['X'])#$coefs
e1x = coef(harmonics[1]['X'])
e2x =coef(harmonics[2]['X'])
e3x =coef(harmonics[3]['X'])
e4x =coef(harmonics[4]['X'])
e1y = coef(harmonics[1]['Y'])
e2y =coef(harmonics[2]['Y'])
e3y =coef(harmonics[3]['Y'])
e4y =coef(harmonics[4]['Y'])
print(dim(mean$coefs))
mx =  mean$coefs[,1,1]
my =  mean$coefs[,1,2]
m = c(mx,my)

#simply take the scores directly from fpca and sum over all dimensions
sampleidx = 3 # select the sample index. there are 20 samples in the dataset


sx = samplecoefs[,sampleidx,1]
sy = samplecoefs[,sampleidx,2]

#simply add the score of the two dimensions
scores = fdarotpcaList$scores
x1 = scores[sampleidx,1,1] + scores[sampleidx,1,2]
x2 = scores[sampleidx,2,1] + scores[sampleidx,2,2]
x3 = scores[sampleidx,3,1] + scores[sampleidx,3,2]
x4 = scores[sampleidx,4,1] + scores[sampleidx,4,2]
print (x1)
print (x2)
print (x3)
print (x4)
#construct the new coefficients using the calculated weights for each eigen function/harmonic

newcoefx = x1*e1x + x2*e2x + x3*e3x+ x4*e4x +mx 
newcoefy = x1*e1y + x2*e2y + x3*e3y+ x4*e4y +my 
newfx = fd(coef =newcoefx ,basisobj = harmonics$basis)
newfy = fd(coef =newcoefy ,basisobj = harmonics$basis)

#plot the function together with the original functions
ox = fd(coef = sx, basisobj=fdafd$basis)
oy = fd(coef = sy, basisobj=fdafd$basis)
plot(ox)
lines(newfx)
plot(oy)
lines(newfy)

#calculate the mean squared error
valsx = eval.fd(seq(0, {numframes-1}, len = {numframes}), newfx)
ovalsx = eval.fd(seq(0, {numframes-1}, len = {numframes}), ox)
valsy = eval.fd(seq(0, {numframes-1}, len = {numframes}), newfy)
ovalsy = eval.fd(seq(0, {numframes-1}, len = {numframes}), oy)
error = 0
for (i in 1:numframes){
  error = error + sqrt( (valsx[i]-ovalsx[i])^2 + (valsy[i]-ovalsy[i])^2  )
}
error = error/numframes
#error = norm(valsx-ovalsx)/2300
print('error')
print(error)
