"""
Sokoban Solver using SAT (Boilerplate)
--------------------------------------
Instructions:
- Implement encoding of Sokoban into CNF.
- Use PySAT to solve the CNF and extract moves.
- Ensure constraints for player movement, box pushes, and goal conditions.

Grid Encoding:
- 'P' = Player
- 'B' = Box
- 'G' = Goal
- '#' = Wall
- '.' = Empty space
"""

from pysat.formula import CNF
from pysat.solvers import Solver

# Directions for movement
DIRS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


class SokobanEncoder:
    def __init__(self, grid, T):
        """
        Initialize encoder with grid and time limit.

        Args:
            grid (list[list[str]]): Sokoban grid.
            T (int): Max number of steps allowed.
        """
        self.grid = grid
        self.T = T
        self.N = len(grid)
        self.M = len(grid[0])

        self.goals = []
        self.boxes = []
        self.player_start = None

        # TODO: Parse grid to fill self.goals, self.boxes, self.player_start
        self._parse_grid()

        self.num_boxes = len(self.boxes)
        self.cnf = CNF()

    def _parse_grid(self):
        """Parse grid to find player, boxes, and goals."""
        # TODO: Implement parsing logic
        for i in range(self.N):
            for j in range(self.M):
                if self.grid[i][j]=='P':
                    self.player_start=(i,j)
                elif self.grid[i][j]=='B':
                    self.boxes.append((i,j))
                elif self.grid[i][j]=="G":
                    self.goals.append((i,j))
                    
        pass

    # ---------------- Variable Encoding ----------------
    def var_player(self, x, y, t):
        """
        Variable ID for player at (x, y) at time t.
        """
        pass self.N*self.M*t+self.M*x+y+1;
        # TODO: Implement encoding scheme

    def var_box(self, b, x, y, t):
        """
        Variable ID for box b at (x, y) at time t.
        """
        # TODO: Implement encoding scheme
        pass t*self.N*self.M+b*self.N+x*self.M+y;

    # ---------------- Encoding Logic ----------------
    def encode(self):
        """
        Build CNF constraints for Sokoban:
        - Initial state
        - Valid moves (player + box pushes)
        - Non-overlapping boxes
        - Goal condition at final timestep
        """
        # TODO: Add constraints for:
        # 1. Initial conditions
        # 2. Player movement
        # 3. Box movement (push rules)
        # 4. Non-overlap constraints
        # 5. Goal conditions
        # 6. Other conditions
        x,y=self.player_start
        
        self.cnf.append([self.var_player(x,y,0)])
        for b in range(len(self.boxes)):
            b_x,b_y=self.boxes[b]
            self.cnf.append([self.var_box(b,b_x,b_y,0)])
        for t in range(self.T+1):
            clause=[]
            for x in range(self.N):
                for y in range(self.M):
                    clause.append(self.var_player(x,y,t))
            self.cnf.append(clause)
        for t in range(self.T+1):
            for x1 in range(self.N):
                for y1 in range(self.M):
                    for x2 in range(self.N):
                        for y2 in range(self.M):
                            if (x1,y1)<(x2,y2):
                                self.cnf.append([-self.var_player(x1,y1,t),-self.var_player(x2,y2,t)])

        for t in range(self.T+1):
            for b in range(len(self.boxes)):
                clause=[]
                for x in range(self.N):
                    for y in range(self.M):
                        clause.append(self.var_box(b,x,y,t))
                self.cnf.append(clause)

        for t in range(self.T+1):
            for x in range(self.N):
                for y in range(self.M):
                    for b1 in range(len(self.boxes)):
                        for b2 in range(b1+1,len(self.boxes)):
                            self.cnf.append([-self.var_box(b1,x,y,t),-self.var_box(b2,x,y,t)])
        for t in range(self.T+1):
            for b in range(len(self.boxes)):
                for x1 in range(self.N):
                    for y1 in range(self.M):
                        for x2 in range(self.N):
                            for y2 in range(self.M):
                                if (x1,y1) < (x2,y2):
                                    self.cnf.append([
                                        -self.var_box(b,x1,y1,t),
                                        -self.var_box(b,x2,y2,t)
                                    ])
        for t in range(self.T+1):
            for b in range(len(self.boxes)):
                for x in range(self.N):
                    for y in range(self.M):
                        self.cnf.append([
                            -self.var_player(x,y,t),
                            -self.var_box(b,x,y,t)
                        ])

def decode(model, encoder):
    """
    Decode SAT model into list of moves ('U', 'D', 'L', 'R').

    Args:
        model (list[int]): Satisfying assignment from SAT solver.
        encoder (SokobanEncoder): Encoder object with grid info.

    Returns:
        list[str]: Sequence of moves.
    """
    N, M, T = encoder.N, encoder.M, encoder.T

    # TODO: Map player positions at each timestep to movement directions
    pass


def solve_sokoban(grid, T):
    """
    DO NOT MODIFY THIS FUNCTION.

    Solve Sokoban using SAT encoding.

    Args:
        grid (list[list[str]]): Sokoban grid.
        T (int): Max number of steps allowed.

    Returns:
        list[str] or "unsat": Move sequence or unsatisfiable.
    """
    encoder = SokobanEncoder(grid, T)
    cnf = encoder.encode()

    with Solver(name='g3') as solver:
        solver.append_formula(cnf)
        if not solver.solve():
            return -1

        model = solver.get_model()
        if not model:
            return -1

        return decode(model, encoder)


