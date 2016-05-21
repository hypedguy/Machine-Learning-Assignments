library("devtools") #libraries
install_github("kassambara/factoextra")
library("factoextra")
######### Read the data #########
faces <- read.csv("faces.csv", header = F)

######### Function that normalizes a vector x (i.e. |x|=1 ) #########
normalize <- function(x) {
  norm = sqrt(sum(x^2))
  x/norm
}

######### Display first face #########

# Useful functions
# > unlist(X)
#   unlist() simplifies the list structure X so that it produces a vector with only atomic elements
#   apply unlist() on a datafram if unable to conduct matrix operations (see code below)
#   for more, see help(unlist)
# > matrix(X,M,N,byrow = F)
#   transforms X into an M-by-N matrix, by default, byrow = F, meaning that the function will use
#   the elements of X to fill an M-by-N matrix by row. If byrow = T, the matrix fill be filled by row

first_face <- matrix(unlist(faces[1,]),64,64)
image(first_face , col=gray((0:255)/255))  # The first input of the image function must be a square matrix


######### Display a random face #########

# Useful functions:
# > sample(1:N,m) 
#   generate m integers from 1 to N


random_face <- matrix(unlist(faces[sample(1:400,1),]), 64, 64) 
image(random_face, col=gray((0:255)/255))


######### Compute and display mean face #########

# Useful functions:
# > colMeans(X) 
#   X can be a matrix or data frame, calculates the mean of X along each column,
#   generate a row vector/datafram

s = vector(,64)
s <- s + colMeans(faces) #generate mean face
image(matrix(unlist(s), 64, 64), col=gray((0:255)/255))

######### Centralize the faces by subtracting the mean face #########

# Useful functions:
# > rep(X,N)
#   replicate a vector X N times
# > as.vector(X)
#   transforms X into a vector

faces_2 <- faces
for(i in faces_2){ #substract the mean face from all faces
	i <- i - s
}

######### Perform PCA #########

# In R, you can compute the eigenvalues and eigenvectors of a square matrix X with the following line of code
# > eigX <- eigen(X)
# In order to obtain the eigenvalues and eigenvectors of B, we do the following
# > eigValX <- eigX$values
# > eigVecX <- eigX$vectors
# where eigValX contrains the eigenvalues of X and eigVecX contains the eigenvectors of X
# For more, see help(eigen)
# Also, recall that in R, matrix multiplication is performed by "%*%", whereas "*" implements element-wise
# multiplication
# The function t(X) gives the transpose of the matrix X (X must be transformed into a matrix first)

irs <- prcomp(faces_2, scale = TRUE) #pca

######### Display first five eigenfaces #########
#display the first 5 eigen vectors as faces
image(matrix(unlist(irs$rotation[,1]), 64, 64), col=gray((0:255)/255))
image(matrix(unlist(irs$rotation[,2]), 64, 64), col=gray((0:255)/255))
image(matrix(unlist(irs$rotation[,3]), 64, 64), col=gray((0:255)/255))
image(matrix(unlist(irs$rotation[,4]), 64, 64), col=gray((0:255)/255))
image(matrix(unlist(irs$rotation[,5]), 64, 64), col=gray((0:255)/255))


######### Reconstruct first face using the first two PCs #########

restr <- irs$x[,1:2] %*% t(irs$rotation[,1:2]) #formulate the reconstruction matrix with 2 pca
image(matrix(unlist(restr[1,]), 64, 64), col=gray((0:255)/255))#show the first image

######### Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs #########

pc <- c(5,10,25,50,100,200,300,399) 
indx <- sample(1:400, 1) #get the id of the random face
for(i in pc){
	restr <- irs$x[,1:i] %*% t(irs$rotation[,1:i]) #formulate the reconstruction matrix
	image(matrix(unlist(restr[indx,]), 64, 64), col=gray((0:255)/255))
}
######### Plot proportion of variance of all the PCs #########

# You can use the following lines of code to plot the vector x
# > plot(x,pch=".")
# > lines(x)
# The x axis will correspond to the indices in the vector.

fviz_eig(irs, addlabels = TRUE, hjust = -0.3) #plot the variance 
plot(summary(irs)$importance[3,], type="l", ylab="%how close to original", xlab="Number of pc used")

