import csv
import os
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
from datetime import datetime

def calculate_position_jump(pos1, pos2):
    """计算两个位置之间的欧氏距离"""
    return np.sqrt(sum((p2 - p1)**2 for p1, p2 in zip(pos1, pos2)))

def calculate_quaternion_distance(q1, q2):
    """计算两个四元数之间的角度距离"""
    # 确保四元数单位化
    q1 = np.array(q1) / np.linalg.norm(q1)
    q2 = np.array(q2) / np.linalg.norm(q2)
    
    # 计算内积
    dot_product = abs(np.dot(q1, q2))
    # 限制在[-1,1]范围内避免数值错误
    dot_product = np.clip(dot_product, 0, 1)
    
    # 计算角度距离
    return 2 * np.arccos(dot_product)

def plot_data_distributions(all_data, output_dir=".", filename_prefix="data_analysis"):
    """使用Plotly绘制数据分布图"""
    if not all_data:
        print("没有数据可以绘制")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个tracker创建图表
    for tracker_name, data in all_data.items():
        if len(data['timestamps']) < 2:
            continue
            
        # 创建子图布局：2行2列
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Abnormal Timestamp Differences (>0.02s)',
                'Position Jump Distribution',
                'Orientation Jump Distribution', 
                '3D Position Trajectory'
            ],
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter3d"}]]
        )
        
        # 计算时间戳差值，只显示超过0.02s的异常值
        time_diffs = np.diff(data['timestamps'])
        abnormal_time_diffs = time_diffs[time_diffs > 0.02]
        
        # 1. 异常时间戳差值分布（只显示>0.02s的）
        if len(abnormal_time_diffs) > 0:
            fig.add_trace(
                go.Histogram(
                    x=abnormal_time_diffs,
                    name='Abnormal Time Diffs',
                    nbinsx=50,
                    opacity=0.7
                ),
                row=1, col=1
            )
            # 添加统计信息到标题
            avg_abnormal = np.mean(abnormal_time_diffs)
            subplot_titles = list(fig.layout.annotations)
            subplot_titles[0].text = f'Abnormal Timestamp Differences (>{0.02}s)<br>Count: {len(abnormal_time_diffs)}, Mean: {avg_abnormal:.4f}s'
        else:
            # 如果没有异常值，显示空图表并说明
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0],
                    mode='text',
                    text=['No abnormal timestamp differences found'],
                    textposition='middle center',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. 位置跳跃分布（三维空间距离）
        positions = np.array(data['positions'])
        if len(positions) > 1:
            position_jumps = [calculate_position_jump(positions[i-1], positions[i]) 
                            for i in range(1, len(positions))]
            fig.add_trace(
                go.Histogram(
                    x=position_jumps,
                    name='Position Jumps',
                    nbinsx=50,
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # 3. 姿态跳跃分布（四元数角度距离）
        orientations = np.array(data['orientations'])
        if len(orientations) > 1:
            orientation_jumps = [calculate_quaternion_distance(orientations[i-1], orientations[i]) 
                               for i in range(1, len(orientations))]
            fig.add_trace(
                go.Histogram(
                    x=orientation_jumps,
                    name='Orientation Jumps',
                    nbinsx=50,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # 4. 3D位置轨迹
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1], 
                z=positions[:, 2],
                mode='lines+markers',
                marker=dict(size=3),
                line=dict(width=2),
                name='3D Trajectory'
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            height=800,
            title_text=f'Data Analysis - {tracker_name}',
            showlegend=False
        )
        
        # 更新x轴标签
        fig.update_xaxes(title_text="Time Difference (s)", row=1, col=1)
        fig.update_xaxes(title_text="Position Jump (m)", row=1, col=2)
        fig.update_xaxes(title_text="Orientation Jump (rad)", row=2, col=1)
        
        # 更新y轴标签
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        # 更新3D场景
        fig.update_scenes(
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            zaxis_title="Z Position (m)",
            row=2, col=2
        )
        
        # 保存图表
        output_file = os.path.join(output_dir, f"{filename_prefix}_{tracker_name}.html")
        plot(fig, filename=output_file, auto_open=False)
        print(f"图表已保存: {output_file}")
        
        # 显示图表
        fig.show()

def check_timestamp_consistency(csv_files, thresholds=None, plot_distributions=True, save_plots=True):
    """
    检查CSV文件中相邻帧的时间戳差值是否超过指定阈值，并检查位置和姿态数据
    
    Args:
        csv_files: 可以是以下格式之一：
                  - 单个文件路径 (str)
                  - 文件路径列表 (list)
                  - 包含文件路径的目录路径 (str)
        thresholds: 阈值字典，包含：
                   - 'timestamp_diff': 时间戳差值阈值，默认0.03秒
                   - 'position_jump': 位置跳跃阈值，默认0.1米
                   - 'orientation_jump': 姿态跳跃阈值，默认0.1弧度
        plot_distributions: 是否绘制数据分布图
        save_plots: 是否保存图表到PDF文件
    
    Returns:
        dict: 包含每个文件检查结果的字典
    """
    # 设置默认阈值
    if thresholds is None:
        thresholds = {
            'timestamp_diff': 0.03,
            'position_jump': 0.1,
            'orientation_jump': 0.1
        }
    
    print(f"\n=== CSV Data Quality Check ===")
    print(f"时间戳差值阈值: {thresholds['timestamp_diff']}s")
    print(f"位置跳跃阈值: {thresholds['position_jump']}m")
    print(f"姿态跳跃阈值: {thresholds['orientation_jump']}rad")
    print("=" * 50)
    
    # 处理输入参数，统一转换为文件路径列表
    file_paths = []
    
    if isinstance(csv_files, str):
        if os.path.isdir(csv_files):
            # 如果是目录，查找所有CSV文件
            for root, dirs, files in os.walk(csv_files):
                for file in files:
                    if file.endswith('.csv'):
                        file_paths.append(os.path.join(root, file))
        elif csv_files.endswith('.csv'):
            # 如果是单个CSV文件
            file_paths.append(csv_files)
        else:
            print(f"错误: {csv_files} 不是有效的CSV文件或目录")
            return {}
    elif isinstance(csv_files, list):
        # 如果是文件列表
        file_paths = [f for f in csv_files if f.endswith('.csv')]
    else:
        print("错误: csv_files 参数必须是文件路径、文件列表或目录路径")
        return {}
    
    if not file_paths:
        print("未找到CSV文件")
        return {}
    
    results = {}
    all_data = {}  # 存储所有数据用于绘图
    
    for csv_file_path in file_paths:
        if not os.path.exists(csv_file_path):
            print(f"文件不存在: {csv_file_path}")
            continue
            
        # 从文件名提取tracker信息
        filename = os.path.basename(csv_file_path)
        if "right" in filename.lower():
            tracker_name = "right"
        elif "left" in filename.lower():
            tracker_name = "left"
        elif "root" in filename.lower():
            tracker_name = "root"
        else:
            tracker_name = filename.replace('.csv', '')
            
        try:
            # 读取CSV文件
            timestamps = []
            positions = []
            orientations = []
            
            with open(csv_file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过标题行
                for row in reader:
                    if len(row) >= 9:  # frame_idx,timestamp,x,y,z,q_x,q_y,q_z,q_w
                        try:
                            timestamp = float(row[1])
                            x, y, z = float(row[2]), float(row[3]), float(row[4])
                            q_x, q_y, q_z, q_w = float(row[5]), float(row[6]), float(row[7]), float(row[8])
                            
                            timestamps.append(timestamp)
                            positions.append([x, y, z])
                            orientations.append([q_x, q_y, q_z, q_w])
                        except ValueError:
                            continue  # 跳过无效的数据行
            
            if len(timestamps) < 2:
                print(f"文件 {filename} ({tracker_name}): 数据不足，无法检查一致性")
                results[csv_file_path] = {
                    'status': 'insufficient_data',
                    'total_frames': len(timestamps),
                    'timestamp_violations': 0,
                    'position_violations': 0,
                    'orientation_violations': 0
                }
                continue
            
            # 存储数据用于绘图
            all_data[tracker_name] = {
                'timestamps': timestamps,
                'positions': positions,
                'orientations': orientations
            }
            
            # 检查时间戳差值
            timestamp_violations = 0
            timestamp_violation_details = []
            max_timestamp_diff = 0
            total_intervals = len(timestamps) - 1
            
            for i in range(1, len(timestamps)):
                diff = timestamps[i] - timestamps[i-1]
                max_timestamp_diff = max(max_timestamp_diff, diff)
                if diff > thresholds['timestamp_diff']:
                    timestamp_violations += 1
                    timestamp_violation_details.append((i, diff))
            
            # 检查位置跳跃
            position_violations = 0
            position_violation_details = []
            max_position_jump = 0
            
            for i in range(1, len(positions)):
                jump = calculate_position_jump(positions[i-1], positions[i])
                max_position_jump = max(max_position_jump, jump)
                if jump > thresholds['position_jump']:
                    position_violations += 1
                    position_violation_details.append((i, jump))
            
            # 检查姿态跳跃
            orientation_violations = 0
            orientation_violation_details = []
            max_orientation_jump = 0
            
            for i in range(1, len(orientations)):
                jump = calculate_quaternion_distance(orientations[i-1], orientations[i])
                max_orientation_jump = max(max_orientation_jump, jump)
                if jump > thresholds['orientation_jump']:
                    orientation_violations += 1
                    orientation_violation_details.append((i, jump))
            
            # 打印结果
            print(f"文件: {filename} ({tracker_name})")
            print(f"  总帧数: {len(timestamps)}")
            print(f"  时间间隔总数: {total_intervals}")
            
            # 时间戳检查结果
            print(f"  最大时间间隔: {max_timestamp_diff:.6f}s")
            print(f"  超过{thresholds['timestamp_diff']}s的间隔数: {timestamp_violations}")
            print(f"  时间戳违规率: {timestamp_violations/total_intervals*100:.2f}%" if total_intervals > 0 else "  时间戳违规率: 0.00%")
            
            # 位置检查结果
            print(f"  最大位置跳跃: {max_position_jump:.6f}m")
            print(f"  超过{thresholds['position_jump']}m的跳跃数: {position_violations}")
            print(f"  位置违规率: {position_violations/total_intervals*100:.2f}%" if total_intervals > 0 else "  位置违规率: 0.00%")
            
            # 姿态检查结果
            print(f"  最大姿态跳跃: {max_orientation_jump:.6f}rad")
            print(f"  超过{thresholds['orientation_jump']}rad的跳跃数: {orientation_violations}")
            print(f"  姿态违规率: {orientation_violations/total_intervals*100:.2f}%" if total_intervals > 0 else "  姿态违规率: 0.00%")
            
            # 总体状态判断
            total_violations = timestamp_violations + position_violations + orientation_violations
            if total_violations == 0:
                print(f"  ✓ 数据质量检查通过")
                status = 'pass'
            else:
                print(f"  ✗ 数据质量检查失败 (总违规: {total_violations})")
                status = 'fail'
                
                # 显示违规详情
                if timestamp_violation_details:
                    print(f"  时间戳违规详情 (前3个):")
                    for idx, (frame_idx, diff_val) in enumerate(timestamp_violation_details[:3]):
                        print(f"    帧 {frame_idx}: {diff_val:.6f}s")
                
                if position_violation_details:
                    print(f"  位置违规详情 (前3个):")
                    for idx, (frame_idx, jump_val) in enumerate(position_violation_details[:3]):
                        print(f"    帧 {frame_idx}: {jump_val:.6f}m")
                
                if orientation_violation_details:
                    print(f"  姿态违规详情 (前3个):")
                    for idx, (frame_idx, jump_val) in enumerate(orientation_violation_details[:3]):
                        print(f"    帧 {frame_idx}: {jump_val:.6f}rad")
            print()
            
            # 保存结果
            results[csv_file_path] = {
                'status': status,
                'total_frames': len(timestamps),
                'total_intervals': total_intervals,
                'timestamp_violations': timestamp_violations,
                'timestamp_violation_rate': timestamp_violations/total_intervals*100 if total_intervals > 0 else 0,
                'max_timestamp_diff': max_timestamp_diff,
                'position_violations': position_violations,
                'position_violation_rate': position_violations/total_intervals*100 if total_intervals > 0 else 0,
                'max_position_jump': max_position_jump,
                'orientation_violations': orientation_violations,
                'orientation_violation_rate': orientation_violations/total_intervals*100 if total_intervals > 0 else 0,
                'max_orientation_jump': max_orientation_jump,
                'tracker_name': tracker_name
            }
            
        except Exception as e:
            print(f"检查文件 {filename} 时出错: {e}")
            results[csv_file_path] = {
                'status': 'error',
                'error': str(e)
            }
    
    # 绘制数据分布图
    if plot_distributions and all_data:
        try:
            plot_data_distributions(all_data)
        except Exception as e:
            print(f"绘制分布图时出错: {e}")
    
    return results


def main():
    """
    示例用法
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='CSV数据质量检查工具')
    parser.add_argument('files', nargs='+', help='CSV文件路径或目录路径')
    parser.add_argument('--timestamp-threshold', type=float, default=0.03, 
                       help='时间戳差值阈值 (默认: 0.03s)')
    parser.add_argument('--position-threshold', type=float, default=0.1, 
                       help='位置跳跃阈值 (默认: 0.1m)')
    parser.add_argument('--orientation-threshold', type=float, default=0.1, 
                       help='姿态跳跃阈值 (默认: 0.1rad)')
    parser.add_argument('--no-plot', action='store_true', 
                       help='不绘制分布图')
    
    # 如果没有参数，显示帮助信息
    if len(sys.argv) == 1:
        print("用法:")
        print("  python csv_check.py <csv文件路径>")
        print("  python csv_check.py <包含csv文件的目录路径>")
        print("  python csv_check.py file1.csv file2.csv file3.csv")
        print("\n可选参数:")
        print("  --timestamp-threshold 0.02    设置时间戳阈值")
        print("  --position-threshold 0.05     设置位置跳跃阈值")
        print("  --orientation-threshold 0.05  设置姿态跳跃阈值")
        print("  --no-plot                     不绘制分布图")
        return
    
    args = parser.parse_args()
    
    # 设置阈值
    thresholds = {
        'timestamp_diff': args.timestamp_threshold,
        'position_jump': args.position_threshold,
        'orientation_jump': args.orientation_threshold
    }
    
    # 执行检查
    if len(args.files) == 1:
        # 单个文件或目录
        results = check_timestamp_consistency(
            args.files[0], 
            thresholds=thresholds, 
            plot_distributions=not args.no_plot
        )
    else:
        # 多个文件
        results = check_timestamp_consistency(
            args.files, 
            thresholds=thresholds, 
            plot_distributions=not args.no_plot
        )
    
    # 打印总结
    if results:
        total_files = len(results)
        passed_files = sum(1 for r in results.values() if r.get('status') == 'pass')
        failed_files = sum(1 for r in results.values() if r.get('status') == 'fail')
        error_files = sum(1 for r in results.values() if r.get('status') == 'error')
        insufficient_data = sum(1 for r in results.values() if r.get('status') == 'insufficient_data')
        
        print("\n" + "=" * 50)
        print("=== 检查总结 ===")
        print(f"总文件数: {total_files}")
        print(f"通过检查: {passed_files}")
        print(f"未通过检查: {failed_files}")
        print(f"数据不足: {insufficient_data}")
        print(f"检查出错: {error_files}")
        
        # 详细违规统计
        if failed_files > 0:
            print("\n违规统计:")
            total_timestamp_violations = sum(r.get('timestamp_violations', 0) for r in results.values())
            total_position_violations = sum(r.get('position_violations', 0) for r in results.values())
            total_orientation_violations = sum(r.get('orientation_violations', 0) for r in results.values())
            
            print(f"  时间戳违规总数: {total_timestamp_violations}")
            print(f"  位置违规总数: {total_position_violations}")
            print(f"  姿态违规总数: {total_orientation_violations}")


if __name__ == "__main__":
    main()