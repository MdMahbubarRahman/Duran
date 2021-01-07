"""
This file contains necessary utility functions required for column generation algorithm.
"""
"""
our problem or model structure is as follows:
min z[1]+z[2]+z[3]
st.
    Ax+By <= 0

    z[1] >= y[1]*L[1]
    z[2] >= y[2]*L[2]
    z[3] >= y[3]*L[3]

    z[1] >= (y[1]-1)*U[1]+∇f1(̄x)(x-̄x)
    z[2] >= (y[2]-1)*U[2]+∇f2(̄x)(x-̄x)
    z[3] >= (y[3]-1)*U[3]+∇f3(̄x)(x-̄x)

    x = [x[1], x[2], x[3]], :Cont
    y = [y[1], y[2], y[3]], :Bin
    z = [z[1], z[2], z[3]], :Cont
    A = (mxn) matrix
    B = (mxn) matrix

First, we need to do dantzig-wolfe decomposition of the above problem .

The constraint set
"Ax+By <= 0"
acts as feasibility constraint and these constraints indirectly influence the objective function.

The constraint set
"z[1] >= y[1]*L[1]
 z[2] >= y[2]*L[2]
 z[3] >= y[3]*L[3]
 z[1] >= (y[1]-1)*U[1]+∇f1(̄x)(x-̄x)
 z[2] >= (y[2]-1)*U[2]+∇f2(̄x)(x-̄x)
 z[3] >= (y[3]-1)*U[3]+∇f3(̄x)(x-̄x)"
directly influence the objective function.

Now we can do Dantzig-Wolfe decomposition of the model as follows:

main problem:
min z[1]+z[2]+z[3]
(x,y) ∈ Q
st.
    z[1] >= y[1]*L[1]
    z[2] >= y[2]*L[2]
    z[3] >= y[3]*L[3]

    z[1] >= (y[1]-1)*U[1]+∇f1(̄x)(x-̄x)
    z[2] >= (y[2]-1)*U[2]+∇f2(̄x)(x-̄x)
    z[3] >= (y[3]-1)*U[3]+∇f3(̄x)(x-̄x)

where Q = {(x,y): Ax+By <= 0, x :Cont, y :Cont}.

The Dantzig-Wolfe relaxation of the main problem:

min z[1]+z[2]+z[3]
(λ,μ)
st.
    z[1] >= sum(u_p[1,i]*L[1]*λ[i] for i in Κ) + sum(u_d[1,j]*L[1]*μ[j] for j in Τ)
    z[2] >= sum(u_p[2,i]*L[2]*λ[i] for i in Κ) + sum(u_d[2,j]*L[2]*μ[j] for j in Τ)
    z[3] >= sum(u_p[3,i]*L[3]*λ[i] for i in Κ) + sum(u_d[3,j]*L[3]*μ[j] for j in Τ)

    z[1]+∇f1(̄x)*̄x+U[1] >= sum((u_p[1,i]*U[1]+∇f1(̄x)*v_p[i])*λ[i] for i in Κ)+sum((u_d[1,j]*U[1]+∇f1(̄x)*v_d[j])*μ[j] for j in Τ)
    z[2]+∇f2(̄x)*̄x+U[2] >= sum((u_p[2,i]*U[2]+∇f2(̄x)*v_p[i])*λ[i] for i in Κ)+sum((u_d[2,j]*U[2]+∇f2(̄x)*v_d[j])*μ[j] for j in Τ)
    z[3]+∇f3(̄x)*̄x+U[3] >= sum((u_p[3,i]*U[3]+∇f3(̄x)*v_p[i])*λ[i] for i in Κ)+sum((u_d[3,j]*U[3]+∇f3(̄x)*v_d[j])*μ[j] for j in Τ)

    sum(λ[i] for i in K) = 1
    λ[i] >= 0, ∀i ∈ K
    μ[j] >= 0, ∀j ∈ T

where (v_p[i], u_p[i]) ≡ (x,y) is the i^th extreme point of Q, and (v_d[j], u_d[j]) is the j^th extreme direction of Q.

Suppose the dual variable solution of the above main relaxed problem is (̄γ,̄π,̄σ).
The pricing subproblem:

max sum(L[i]*y[i]*̄γ[i] for i in 1:3)+sum((U[j]*y[j]+∇f[j](̄x)*x)*̄π[j] for j in 1:3)+̄σ
(x,y)
st. Ax+By <= 0
    x :Cont
    y :Cont
"""
