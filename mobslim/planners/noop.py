from mobslim.planners.core import BasePlanner

class NoopPlanner(BasePlanner):
    def __init__(self, plans: dict, agents: dict):
        self.plans = plans
    
    def update(self, events: list, assets: dict):
        return
    
    def replan(self) -> dict:
        return self.plans