#!/usr/bin/env python3
"""
相机插值图绘制脚本
显示6个原始相机位置和12个插值位置，总共18个Pseudo GT图像
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def create_camera_interpolation_diagram():
    """创建相机插值图"""
    
    # 设置字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 设置坐标轴范围
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect('equal')
    
    # 绘制虚线圆圈
    circle = plt.Circle((0, 0), 0.5, fill=False, linestyle='--', color='gray', linewidth=8)
    ax.add_patch(circle)
    
    # 定义原始6个相机角度（度）- 确保F 0°在最上方
    original_angles = [0, 55, 110, 180, -110, -55]  # Front, FL, BL, Back, BR, FR
    original_labels = ['F', 'FL', 'BL', 'B', 'BR', 'FR']
    
    # 定义插值角度（每两个原始角度之间插值3个）
    interpolated_angles = [
        # F(0°) 到 FL(55°) 之间
        13.8, 27.5, 41.2,
        # FL(55°) 到 BL(110°) 之间  
        68.8, 82.5, 96.2,
        # BL(110°) 到 B(180°) 之间
        127.5, 145.0, 162.5,
        # B(180°) 到 BR(-110°) 之间
        -162.5, -145.0, -127.5,
        # BR(-110°) 到 FR(-55°) 之间
        -96.2, -82.5, -68.8,
        # FR(-55°) 到 F(0°) 之间
        -41.2, -27.5, -13.8
    ]
    
    # 绘制原始6个相机位置（绿色大圆点）
    for i, angle in enumerate(original_angles):
        # 转换为弧度
        rad = np.radians(angle)
        x = np.cos(rad) * 0.5
        y = np.sin(rad) * 0.5
        
        # 绘制绿色大圆点
        circle = plt.Circle((x, y), 0.02, color='green', alpha=0.8)
        ax.add_patch(circle)
        
        # 添加标签（直接文字，无框）
        label_x = x * 1.2
        label_y = y * 1.2
        
        # 添加文本
        ax.text(label_x, label_y, f'{original_labels[i]} {angle}°', 
               ha='center', va='center', fontsize=20, fontweight='bold', color='green')
    
    # 绘制插值位置（紫色小圆点）
    for i, angle in enumerate(interpolated_angles):
        # 转换为弧度
        rad = np.radians(angle)
        x = np.cos(rad) * 0.5
        y = np.sin(rad) * 0.5
        
        # 绘制紫色小圆点
        circle = plt.Circle((x, y), 0.01, color='purple', alpha=0.8)
        ax.add_patch(circle)
        
        # 添加标签（直接文字，无框）
        label_x = x * 1.2
        label_y = y * 1.2
        
        # 添加文本（只显示角度）
        ax.text(label_x, label_y, f'{angle}°', 
               ha='center', va='center', fontsize=19, fontweight='bold', color='purple')
    
    # 在圆心添加vehicle标签
    ax.text(0, 0, 'Vehicle', ha='center', va='center', fontsize=56, fontweight='bold', color='black')
    
    # 添加图例
    # 原始相机图例
    ax.scatter([], [], c='green', s=100, alpha=0.8, label='Original 6-Camera Images')
    # 插值相机图例  
    ax.scatter([], [], c='purple', s=50, alpha=0.8, label='Generated Pseudo GT Images')
    
    ax.legend(loc='center', fontsize=18, bbox_to_anchor=(0.5, 0.4))
    
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('camera_interpolation_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('camera_interpolation_diagram.pdf', bbox_inches='tight')
    plt.show()
    
    print("相机插值图已生成并保存为 camera_interpolation_diagram.png 和 .pdf")

if __name__ == "__main__":
    create_camera_interpolation_diagram() 