# QuantumProject

## Caylay Path
The implementation of the algorithm 'Quantum Supremacy and random circuits'.

Main program is in simulation.py.

MLbased.py is an algorithm based on gradient descent, and the loss is $l_2$ loss, but it does not show good performance as expected, since the loss will converge in many local minimal.

BWAlgo.py is the Berlekamp Welch algorithm in the paper. The linear system solving now use Moore-Penrose pseudoinverse. For linear system $Ax=b$, $A^+b$ will be (one of) its solutions. For the case there is no solution, $z=A^+b$ minimizes $||Az-b||_2$
