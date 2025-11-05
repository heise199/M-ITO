"""
主入口文件 - 运行不同的测试案例
"""
from iga_top2d import iga_top2d

def case():
    """运行不同的拓扑优化案例"""
    # Cantilever beam
    # iga_top2d(10, 5, [1, 1], [161, 81], 1, 0.2, 3, 2)
    
    # MBB beam
    # iga_top2d(18, 3, [1, 1], [241, 41], 2, 0.2, 3, 2)
    
    # Michell-type structure
    iga_top2d(10, 4, [1, 1], [101, 41], 3, 0.2, 3, 2)
    
    # L beam
    # iga_top2d(10, 5, [1, 1], [101, 51], 4, 0.3, 3, 2)
    
    # quarter annulus
    # iga_top2d(10, 10, [0, 1], [101, 51], 5, 0.4, 3, 2)
    
    # test
    # iga_top2d(10, 5, [0, 1], [21, 21], 5, 0.4, 3, 2)

if __name__ == '__main__':
    case()

