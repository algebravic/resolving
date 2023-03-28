Use MAXSAT to find minimal resolving sets of a graph
====================================================
Let :math:`G` be a connected undirected graph, and let :math:`d(u,v)`
denote the minimum distance between nodes :math:`u` and :math:`v`.  We
say that a vertex :math:`x \in V(G)` *resolves* the pair :math:`(u,v)
\in V(G) \times V(G)` if :math:`d(u,x) \ne d(v,x)`.  A *resolving set*
for :math:`G` is a subset :math:`S \subseteq V(G)` such that every
pair of distinct vertices is resolved by some element of :math:`S`.
We would like to find a minimum cardinality resolving set.  We may use
MAXSAT for this.  Define variables :math:`x_s` for all :math:`s \in
V(G)` as an indicator of being in the minimal resolving set.  Define
sets :math:`A_{u,v} = \{s \in V(G): s \text{ resolves } (u,v)\}`.
Then the clauses are :math:`\bigvee_{x \in A_{u,v} x_s`.  We may also
want to cut down the search by using automorphisms of :math:`G`.
Because automorphisms leave the distance invariant, if :math:`S` is a
resolving set, and :math:`\sigma \in \text{Aut}(G)` then
:math:`\sigma(S)` is also resolvong. Let :math:`R \subset
\text{Aut}(G)`.  Fix a total order on subsets of :math:`V(G)` (such as
lexicographic order on the indicator vector). We would like any
solution to satisfy :math:`S \le \sigma(S)` for all :math:`\sigma \in
R`.
