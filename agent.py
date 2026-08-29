import random
import math
from collections import deque
import heapq

from logic_engine import KnowledgeBase


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
        self.active_algo = 'BFS'   # 'BFS', 'DFS', 'UCS', or 'AStar'

        # Practical 04 - Step 3.1: Knowledge Base + safety rules
        self.kb = KnowledgeBase()
        self.kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
        self.kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

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
    # Practical 04 - Step 1.1: Heuristic Functions
    # ------------------------------------------------------------------
    def manhattan_distance(self, pos, goal):
        """h(n) = |x1 - x2| + |y1 - y2|"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        """h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)"""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    # ------------------------------------------------------------------
    # Practical 04 - Step 1.2: A* Search
    # f(n) = g(n) + h(n)
    # ------------------------------------------------------------------
    def _get_tile_facts(self, pos, threats):
        """
        Practical 04 - Step 3.2: Translate the percepts for a specific tile
        into facts for the Knowledge Base.

        NOTE: The current grid game does not emit real 'TargetVisible' /
        'HasDust' / 'BloodseekerMissing' percepts, so this is a literal,
        minimal wiring of the rule inputs described in the practical:
        the agent always 'HasDust' and has no backup ('BloodseekerMissing'
        is always true), and 'TargetVisible' becomes true only when the
        tile coincides with a known threat position. Net effect: any tile
        occupied by a threat chains all the way to 'Retreat' and gets
        marked Infeasible. Adjust this mapping if your game exposes
        richer percepts.
        """
        facts = ['HasDust', 'BloodseekerMissing']
        if threats and pos in threats:
            facts.append('TargetVisible')
        return facts

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan', threats=None):
        width, height = grid_size
        walls = set(tuple(w) for w in walls)
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)
        if start_pos == goal_pos:
            return []

        heuristic_func = self.manhattan_distance if heuristic_type == 'manhattan' else self.euclidean_distance

        reached_states = set()
        counter = 0  # tie-breaker so heapq never compares list/tuple paths directly

        g_start = 0
        h_start = heuristic_func(start_pos, goal_pos)
        f_start = g_start + h_start
        frontier = [(f_start, g_start, counter, start_pos, [])]

        while frontier:
            f_cost, g_cost, _, pos, path = heapq.heappop(frontier)

            if pos == goal_pos:
                return path

            if pos in reached_states:
                continue
            reached_states.add(pos)

            for action, (dx, dy) in self.moves.items():
                new_pos = (pos[0] + dx, pos[1] + dy)
                if 0 <= new_pos[0] < width and 0 <= new_pos[1] < height \
                        and new_pos not in walls and new_pos not in reached_states:

                    # --- Practical 04 - Step 3.2: Knowledge Base feasibility check ---
                    self.kb.clear_facts()
                    for fact in self._get_tile_facts(new_pos, threats):
                        self.kb.tell_fact(fact)
                    self.kb.forward_chain()
                    if 'Retreat' in self.kb.facts:
                        continue  # Reachable (no wall) but logically Infeasible - skip it

                    g_new = g_cost + 1
                    h_new = heuristic_func(new_pos, goal_pos)
                    f_new = g_new + h_new
                    counter += 1
                    heapq.heappush(frontier, (f_new, g_new, counter, new_pos, path + [action]))
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
            elif self.active_algo == 'AStar':
                threats = percept.get('opponents')
                path = self.astar_search(agent_pos, goal_pos, walls, grid_size, threats=threats)
            else:
                path = self.bfs_search(agent_pos, goal_pos, walls, grid_size)

            self.plan = path if path else [random.choice(list(self.moves.keys()))]

        return self.plan.pop(0)


if __name__ == '__main__':
    # Practical 04 - Step 1.1 Testing Checkpoint
    _agent = SearchAgent()
    _start, _goal = (0, 0), (3, 4)
    print("Manhattan distance:", _agent.manhattan_distance(_start, _goal))  # expected 7
    print("Euclidean distance:", _agent.euclidean_distance(_start, _goal))  # expected 5.0
