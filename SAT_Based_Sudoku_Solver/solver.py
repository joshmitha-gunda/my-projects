"""
sudoku_solver.py

Implement the function solve_sudoku(grid: List[List[int]]) -> List[List[int]] using a SAT solver from PySAT.
"""

from pysat.formula import CNF
from pysat.solvers import Solver
from typing import List
def number(x,y,z)->int:
    num=x*10*10 +y*10+z
    return num

def solve_sudoku(grid: List[List[int]]) -> List[List[int]]:
    """Solves a Sudoku puzzle using a SAT solver. Input is a 2D grid with 0s for blanks."""
    cnf=CNF()
    for i in range(1,10):
        for j in range(1,10):
            clause=[]
            for k in range(1,10):
                clause.append(number(i,j,k))
            cnf.append(clause)
            for k in range(1,10):
                for a in range(1,k):
                        cnf.append([-number(i,j,k),-number(i,j,a)])
    for j in range(1,10):
        for k in range(1,10):
            clause=[]
            for i in range(1,10):
                clause.append(number(i,j,k))
            cnf.append(clause)
            for m in range(1,10):
                for n in range(1,m):
                    cnf.append([-number(m,j,k),-number(n,j,k)])
    for i in range(1,10):   
        for k in range(1,10):
            clause=[]
            for j in range(1,10):
                clause.append(number(i,j,k))
            cnf.append(clause)
            for m in range(1,10):
                for n in range(1,m):
                    cnf.append([-number(i,m,k),-number(i,n,k)])
    for k in range(1,10):
        cnf.append([number(1,1,k),number(1,2,k),number(1,3,k),number(2,1,k),number(2,2,k),number(2,3,k),number(3,1,k),number(3,2,k),number(3,3,k)])
        cnf.append([number(4,1,k),number(4,2,k),number(4,3,k),number(5,1,k),number(5,2,k),number(5,3,k),number(6,1,k),number(6,2,k),number(6,3,k)])
        cnf.append([number(7,1,k),number(7,2,k),number(7,3,k),number(8,1,k),number(8,2,k),number(8,3,k),number(9,1,k),number(9,2,k),number(9,3,k)])
        cnf.append([number(1,4,k),number(1,5,k),number(1,6,k),number(2,4,k),number(2,5,k),number(2,6,k),number(3,4,k),number(3,5,k),number(3,6,k)])
        cnf.append([number(4,4,k),number(4,5,k),number(4,6,k),number(5,4,k),number(5,5,k),number(5,6,k),number(6,4,k),number(6,5,k),number(6,6,k)])
        cnf.append([number(7,4,k),number(7,5,k),number(7,6,k),number(8,4,k),number(8,5,k),number(8,6,k),number(9,4,k),number(9,5,k),number(9,6,k)])
        cnf.append([number(1,7,k),number(1,8,k),number(1,9,k),number(2,7,k),number(2,8,k),number(2,9,k),number(3,7,k),number(3,8,k),number(3,9,k)])
        cnf.append([number(4,7,k),number(4,8,k),number(4,9,k),number(5,7,k),number(5,8,k),number(5,9,k),number(6,7,k),number(6,8,k),number(6,9,k)])
        cnf.append([number(7,7,k),number(7,8,k),number(7,9,k),number(8,7,k),number(8,8,k),number(8,9,k),number(9,7,k),number(9,8,k),number(9,9,k)])
        for i in range(1,10):
            for j in range(1,10):
                for m in range (1,i+1):
                    for n in range(1,j+1):
                        if(not (i==j and m==n)):
                            cnf.append([-number(i,j,k),-number(m,n,k)])
    for i in range(0,9):
        for j in range(0,9):
            if( grid[i][j]):
                value=grid[i][j]
                cnf.append([number(i+1,j+1,value)])
    with Solver(name='glucose3') as solver:
        solver.append_formula(cnf.clauses)
        if solver.solve():
            model = solver.get_model()
            for n in model:
                if(n>0):
                    x=n%10
                    n=n//10
                    y=n%10
                    z=n//10
                    grid[x-1][y-1]=z
    return grid

    # TODO: implement encoding and solving using PySAT
    pass
