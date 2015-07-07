# code for the fpca based on the book Functional Data Analysis with Matlab and R 
# approximate the original data via a weighted linear combination of eigenfunctions
# with the weights derived by the lm function
library(fda)

#load the data
load('handwrit.rda')
print(dim(handwrit))
#create smooth bivariate function representation for each replication
fdarange = c(0, 2300)
fdabasis = create.bspline.basis(fdarange, 105, 6)#create a b-spline basis system with 105 functions with degree 6 on equally spaced knots
fdatime = seq(0, 2300, len=1401)
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
addedscore= fdarotpcaList$scores[1,1,1] + fdarotpcaList$scores[1,1,2]
print (addedscore)
#plot.pca.fd(fdarotpcaList)

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
e1 = c(e1x,e1y)
e2 = c(e2x,e2y)
e3 = c(e3x,e3y)
e4 = c(e4x,e4y)
print(dim(mean$coefs))
mx =  mean$coefs[,1,1]
my =  mean$coefs[,1,2]
m = c(mx,my)

#do linear model regression for the stacked xy coeeficient vectors
sampleidx = 5 # select the sample index. there are 20 samples in the dataset
sx = samplecoefs[,sampleidx,1]
sy = samplecoefs[,sampleidx,2]
s = c(sx,sy)
centered = s - m #substract the mean from the sample
x = lm(centered ~ e1 + e2 + e3 + e4)
print(x)

#construct the new coefficients using the calculated weights for each eigen function/harmonic
x1 = x$coef['e1']
x2 = x$coef['e2']
x3 = x$coef['e3']
x4 = x$coef['e4']

print (x1)
print (x2)
print (x3)
print (x4)
intercept = x$coef[1] #use also the estimated intercept
newcoefx = x1*e1x + x2*e2x + x3*e3x+ x4*e4x +mx + intercept
newcoefy = x1*e1y + x2*e2y + x3*e3y+ x4*e4y +my + intercept
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