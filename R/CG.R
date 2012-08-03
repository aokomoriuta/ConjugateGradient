N <- 10

A <- matrix(0, nrow=N, ncol=N)
x <- numeric(N)
b <- numeric(N)

for (i in 0:(N-2))
{
	for (j in (i+1):(N-1))
	{
		a_ij <-abs(sin(i*i + 2*j))

		A[i+1, j+1] <- a_ij
		A[j+1, i+1] <- a_ij

		A[i+1,i+1] <- A[i+1,i+1] + a_ij
		A[j+1,j+1] <- A[j+1,j+1] + a_ij
	}

	x[i+1] <- 0

	b[i+1] <- i*0.2
}
print(A)
print(b)

Ap <- A %*% x
p <- b - Ap
r <- p

print(r)

rr <- 1

while(rr > 1e-12)
{
	rr <- (t(r) %*% r)[1,1]
	Ap <- A %*% p
	alpha = rr / (t(p) %*% Ap)[1,1]

	x <- x + alpha*p
	r <- r - alpha*Ap
	rrNew <- (t(r) %*% r)[1,1]

	print(rrNew)

	beta <- rrNew / rr
	p <- r + beta*p
}

print(x)

solve(A, b)