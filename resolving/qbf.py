"""
Make a QBF model for the hypercube metric dimension.

Plan:

To certify a lower bound of B, construct the following model

forall X[i,j] i in range(B-1), j in range(N)
This corresponds to a putative resolving set.
There Exists Y[j], Z[j], j in range(N).
This corresponds to a pair (y,z) which are *not* resolved by the
putative resolving set.

Define U[i,j] = X[i,j] xor Y[j], for i,j
       V[i,j] = X[i,j] xor Z[j]

We have constraints \/_i X[i] \/ \/_i Y[i] (rule out 0)
                    ~X[i]\/~Y[i] for all i (not simulatenously 1)
                    sum_j U[i,j] + sum_j ~V[i,j] = N for all i

We also need symmetry breaking constaints for the X[i,j]:

sum_j X[i,j] <= sum_j X[i+1, j] for i in range(B-2)
X[0,j] = 0 (the all 0 is in the set).

if sum_j X[i,j] == sum_j X[i+1,j] then X[i] < (lex) X[i+1]

For 0 <= i < k < N, 0 <= r < B-1, E[i,k,r] is true
if and only if X[s,i] == X[s,k] for 0 <= s <= r.

Then we want E[i,k,r] ==> ~(X[r,i] == 0 /\ X[r,k] == 1)
(this is the prime symmetry breaking).

Inductively, we have E[i,k,0] = 1 for all i,k
E[i,k, r + 1] = E[i,k,r] /\ (X[r,i] == X[r,k])

"""
pass
