from mobslim.network import Grid


def test_grid_network():
    network = Grid(
        size=3,
        length=50,  # meters
        lanes=1,
        freespeed=10,  # m/s
        flow_capacity=0.25,  # vehicles per second
    )
    assert network.get_start() == (0, 0)
    assert network.get_end() == (3, 3)