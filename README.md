# QuantumProject

## Caylay Path
The implementation of the algorithm in 'Quantum Supremacy and random circuits'.

Main program is in simulation.py.

MLbased.py is an algorithm based on gradient descent, and the loss is L2 loss, but it does not show good performance as expected, since the loss will converge in many local minimal.

BWAlgo.py is the Berlekamp Welch algorithm in the paper. The linear system solving now use Moore-Penrose pseudoinverse, since in the lecture notes/ introductions about the BW Algoithm don't have a clear discription either. For linear system Ax=b, A^+b will be (one of) its solutions. For the case there is no solution, z=A^+b minimizes ||Az-b||_2. Now the program can run in no error case, but if we introduce measurement error, then the program will have problems.

TODO List: Change the pseudoinverse to other algorithms.(Can check up https://github.com/j2kun/welch-berlekamp). Investigate the performance of BW Algorithm in erroneous cases. 
Rewrite in cirq package.
