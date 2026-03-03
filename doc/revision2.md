
## Reviewer #2:

 The author prepared a careful revision. With the changes, the manuscript is in a much better shape and presents a nice improvement over the state-of-the-art, giving a new perspective on the efficient implementation of the ghost penalty method. The experimental section is well-described and gives a good impression of the method.

### There are a few minor issues left to be checked and possibly fixed:

- **DONE** Please clarify the derivatives in Eq. (33) more clearly. The text two lines after (33) uses \nabla_x and \nabla_\xi explicitly, but on the right-hand side in (33) it is not immediately clear that it is \nabla_\xi, whereas it is \nabla_x on the left hand side. (If I understand the text in the new version correctly.)
- **DONE** Text before formula (34) uses x_s (small s), formula uses x_S. 
- **DONE** Eq. (35) states J^{-1}, whereas Eq. (33), which involves a similar statement, uses J^{-T}. 
- **DONE** Also make sure that the notation for derivatives (\nabla_x or \nabla_xi) is consistent to the previous formula.
- **DONE** Eq. (36) and the respective line 28 in Alg 1 also needs to make sure to use consistent notation for derivatives.
- **DONE** The text that introduces the load balancing in Sec 2.7, part starting with "For load balancing we use a similar strategy to [10]...". It could help the reader to state clearly that the paragraph is about computing in parallel with MPI, using a partitioning of the domain. There is a text explaining parallelization later in this subsection, the paragraph starting with "We use the distributed computing via MPI through infrastructure implemented in deal.II. The cells are distributed among the processors and the operator is applied in parallel...." with some remark in the highlighted changes part that something has been moved. I suggest to reflect on the best possible order to give full information.
- **DONE** In algorithm 1 there is an empty line 8, which is supposed to show gradients at quadrature points.
- [??] In algorithm 1 line 12 information gets queued at quadrature points

[??] Meaning of that point remains unclear to me, I don't see what change is needed.


## Other changes:
- Code availablity statement: link to the GitHub Repo.