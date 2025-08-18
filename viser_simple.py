#!/usr/bin/env python3
"""
超简单的 viser 测试 - 只有最基本的元素
"""

import time
import numpy as np
import viser

def main():
    # 创建服务器
    server = viser.ViserServer(port=8080)
    print("Simple viser server started at http://localhost:8080")
    
    # 只添加一个大球体
    sphere = server.scene.add_icosphere(
        "/big_sphere",
        radius=1.0,  # 大球体
        color=(1.0, 0.0, 0.0)  # 红色
    )
    
    # 球体在原点
    sphere.position = np.array([0.0, 0.0, 0.0])
    
    print("Big red sphere created at origin")
    print("Open http://localhost:8080 in your browser")
    print("You should see a large red sphere")
    print("Use mouse to rotate the view")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
