N <- 21
M <- 6

A <- matrix(0, nrow=N, ncol=N)
x <- numeric(N)
b <- numeric(N)

for (i in 0:(N-1))
{
	for (j in (max(0, i - M/2+1)):(min(N, i + M/2)-1))
	{
		if(i != j)
		{
			a_ij <-abs(sin(i + j))
			A[i+1, j+1] <- a_ij

			A[i+1,i+1] <- A[i+1,i+1] + a_ij
		}
	}

	x[i+1] <- i/10

	b[i+1] <- cos(i) * 10
}
print(A)
#hist(eigen(A)$values)
#kappa(A)

i <- floor((which(A > 0)-1)/N)
j <- (which(A > 0)-1)%%N

plot(j, i)

solve(A, b)



Ap <- A %*% x
p <- b - Ap
r <- p

residual <- 1

#while(residual > 1e-4)
{
	rr <- (t(r) %*% r)[1,1]
	Ap <- A %*% p
	alpha = rr / (t(p) %*% Ap)[1,1]
	x <- x + alpha*p
	r <- r - alpha*Ap
	residual = max(abs(r))
	print(r)


	rrNew <- (t(r) %*% r)[1,1]
	beta <- rrNew / rr
	p <- r + beta*p
}

print(x)
