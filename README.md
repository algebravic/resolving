Use MAXSAT to find minimal resolving sets of a graph
====================================================
Let $G$ be a connected undirected graph, and let $d(u,v)$ denote the
minimum distance between nodes $u$ and $v$.  We say that a vertex $x
\in V(G)$ *resolves* the pair $(u,v) \in V(G) \times V(G)$ if $d(u,x)
\ne d(v,x)$.  A *resolving set* for $G$ is a subset $S \subseteq V(G)$
such that every pair of distinct vertices is resolved by some element
of $S$.  We would like to find a minimum cardinality resolving set.
We may use MAXSAT for this.  Define variables $`x_s`$ for all
$`s \in V(G)`$ as an indicator of being in the minimal resolving set.  Define
sets $`A_{u,v} = \{s \in V(G): s \text{ resolves } (u,v)\}`$.  Then the
clauses are $`\bigvee_{x \in A _ {u,v} x_s}`$.  We may also want to cut
down the search by using automorphisms of $G$.  Because automorphisms
leave the distance invariant, if $S$ is a resolving set, and $`\sigma
\in \text{Aut}(G)`$ then $\sigma(S)$ is also resolving. Let $R \subset
\text{Aut}(G)$.  Fix a total order on subsets of $V(G)$ (such as
lexicographic order on the indicator vector). We would like any
solution to satisfy $S \le \sigma(S)$ for all $\sigma \in R$.
