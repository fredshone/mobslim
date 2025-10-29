class BasePlanner:
    def __init__(self, plans: dict, agents: dict):
        raise NotImplementedError
    
    def update(self, events: list, assets: dict):
        raise NotImplementedError
    
    def plan(self):
        raise NotImplementedError
    
    def replan(self) -> dict:
        raise NotImplementedError