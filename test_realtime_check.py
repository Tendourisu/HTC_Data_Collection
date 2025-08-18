#!/usr/bin/env python3
"""
测试实时数据一致性检查功能
"""

import time
import numpy as np
from ReadOutTrackerForUMICheckOnline import CSVLogger

def simulate_real_time_check():
    """
    模拟实时数据检查功能
    """
    print("=== 实时数据一致性检查测试 ===")
    
    # 创建CSV Logger实例
    csv_logger = CSVLogger()
    
    # 模拟tracker数据
    tracker_number = 0
    base_time = time.time()
    base_position = [0.0, 0.0, 0.0]
    
    print(f"时间戳阈值: {csv_logger.timestamp_threshold}s")
    print(f"位置跳跃阈值: {csv_logger.position_threshold}m")
    print("开始模拟数据流...")
    print()
    
    # 模拟正常数据流
    for i in range(5):
        current_time = base_time + i * 0.008  # 8ms间隔 (正常)
        position = [
            base_position[0] + i * 0.001,  # 小位移
            base_position[1] + i * 0.001,
            base_position[2] + i * 0.001,
            0.0, 0.0, 0.0, 1.0  # 四元数
        ]
        
        print(f"正常数据 {i+1}: 时间={current_time:.6f}, 位置=[{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
        csv_logger.check_real_time_consistency(tracker_number, i, current_time, position)
        
        # 更新前一帧数据
        csv_logger.previous_data[tracker_number] = [current_time, position[0], position[1], position[2]]
        time.sleep(0.1)  # 暂停100ms以便观察
    
    print("\n--- 注入异常数据 ---")
    
    # 模拟时间戳异常
    abnormal_time = base_time + 5 * 0.008 + 0.1  # 100ms延迟 (异常)
    position = [
        base_position[0] + 5 * 0.001,
        base_position[1] + 5 * 0.001,
        base_position[2] + 5 * 0.001,
        0.0, 0.0, 0.0, 1.0
    ]
    
    print(f"异常时间戳数据: 时间={abnormal_time:.6f}, 位置=[{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
    csv_logger.check_real_time_consistency(tracker_number, 5, abnormal_time, position)
    csv_logger.previous_data[tracker_number] = [abnormal_time, position[0], position[1], position[2]]
    time.sleep(0.5)
    
    # 模拟位置异常
    normal_time = abnormal_time + 0.008
    abnormal_position = [
        base_position[0] + 0.1,  # 10cm跳跃 (异常)
        base_position[1] + 0.1,
        base_position[2] + 0.1,
        0.0, 0.0, 0.0, 1.0
    ]
    
    print(f"异常位置数据: 时间={normal_time:.6f}, 位置=[{abnormal_position[0]:.4f}, {abnormal_position[1]:.4f}, {abnormal_position[2]:.4f}]")
    csv_logger.check_real_time_consistency(tracker_number, 6, normal_time, abnormal_position)
    time.sleep(0.5)
    
    print("\n测试完成!")

if __name__ == "__main__":
    simulate_real_time_check()
