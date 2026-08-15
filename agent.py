import random
from collections import deque
import heapq


class GreedyGridAgent:
    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        pos = percept['agent_pos']
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here', False):
            return 'Up'
        if percept.get('wall_ahead', False):
            return random.choice(self.actions_pool)
        return 'Up'


class ModelBasedAgent:
    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']
        self.last_action = None
        self.visited_cells = set()

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('wall_ahead', False):
            choices = [a for a in self.actions_pool if a != self.last_action]
            action = random.choice(choices)
        elif percept.get('food_here', False):
            action = 'Up'
        else:
            action = 'Up'
        self.last_action = action
        return action


class SearchAgent:
    def __init__(self):
        self.moves = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}
        # Practical 03 additions
        self.plan = []
        self.active_algo = 'BFS'   # 'BFS', 'DFS', or 'UCS'

    # ------------------------------------------------------------------
    # Breadth-First Search (unchanged from Practical 03 starter code)
    # ------------------------------------------------------------------
    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        walls = set(tuple(w) for w in walls)
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)
        if start_pos == goal_pos:
            return []
        visited = {start_pos}  # reached set -> Graph Search
        queue = deque([(start_pos, [])])
        while queue:
            pos, path = queue.popleft()
            for action, (dx, dy) in self.moves.items():
                new_pos = (pos[0] + dx, pos[1] + dy)
                if 0 <= new_pos[0] < width and 0 <= new_pos[1] < height \
                        and new_pos not in walls and new_pos not in visited:
                    new_path = path + [action]
                    if new_pos == goal_pos:
                        return new_path
                    visited.add(new_pos)
                    queue.append((new_pos, new_path))
        return None

    # ------------------------------------------------------------------
    # Depth-First Search (LIFO stack, list.pop())
    # ------------------------------------------------------------------
    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        walls = set(tuple(w) for w in walls)
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)
        if start_pos == goal_pos:
            return []
        visited = {start_pos}  # reached set -> Graph Search
        stack = [(start_pos, [])]
        while stack:
            pos, path = stack.pop()
            if pos == goal_pos:
                return path
            for action, (dx, dy) in self.moves.items():
                new_pos = (pos[0] + dx, pos[1] + dy)
                if 0 <= new_pos[0] < width and 0 <= new_pos[1] < height \
                        and new_pos not in walls and new_pos not in visited:
                    visited.add(new_pos)
                    stack.append((new_pos, path + [action]))
        return None

    # ------------------------------------------------------------------
    # Uniform-Cost Search (priority queue via heapq, ordered by g(n))
    # ------------------------------------------------------------------
    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        walls = set(tuple(w) for w in walls)
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)
        if start_pos == goal_pos:
            return []

        counter = 0  # tie-breaker so heapq never compares list/tuple paths directly
        frontier = [(0, counter, start_pos, [])]
        best_cost = {start_pos: 0}  # reached set with best known g(n)

        while frontier:
            cost, _, pos, path = heapq.heappop(frontier)

            if pos == goal_pos:
                return path

            if cost > best_cost.get(pos, float('inf')):
                continue  # stale entry, a cheaper path already expanded this node

            for action, (dx, dy) in self.moves.items():
                new_pos = (pos[0] + dx, pos[1] + dy)
                if 0 <= new_pos[0] < width and 0 <= new_pos[1] < height and new_pos not in walls:
                    new_cost = cost + 1  # every step has uniform cost of 1
                    if new_pos not in best_cost or new_cost < best_cost[new_pos]:
                        best_cost[new_pos] = new_cost
                        counter += 1
                        heapq.heappush(frontier, (new_cost, counter, new_pos, path + [action]))
        return None

    # ------------------------------------------------------------------
    # Offline planning: build a full plan once, then execute it action by action
    # ------------------------------------------------------------------
    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            agent_pos = tuple(percept.get('agent_pos', (0, 0)))
            all_food = [tuple(f) for f in percept.get('all_food', [])]
            walls = percept.get('walls', [])
            grid_size = percept.get('grid_size', (10, 10))

            if not all_food:
                return random.choice(list(self.moves.keys()))

            # Find the closest food pellet (Manhattan distance) as the search goal
            goal_pos = min(all_food, key=lambda f: self._manhattan(agent_pos, f))

            if self.active_algo == 'BFS':
                path = self.bfs_search(agent_pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'DFS':
                path = self.dfs_search(agent_pos, goal_pos, walls, grid_size)
            elif self.active_algo == 'UCS':
                path = self.ucs_search(agent_pos, goal_pos, walls, grid_size)
            else:
                path = self.bfs_search(agent_pos, goal_pos, walls, grid_size)

            self.plan = path if path else [random.choice(list(self.moves.keys()))]

        return self.plan.pop(0)
