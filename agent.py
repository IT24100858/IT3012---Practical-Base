import random
from collections import deque


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

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        walls = set(tuple(w) for w in walls)
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)

        if start_pos == goal_pos:
            return []

        visited = {start_pos}
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
