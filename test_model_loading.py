#!/usr/bin/env python3
"""
测试stable-diffusion-2-base模型加载
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os

def test_model_loading():
    print("🚀 开始测试模型加载...")
    
    # 检查CUDA可用性
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name()}")
    
    # 模型路径
    model_path = "/home/lyt/stable-diffusion-2-base"
    
    print(f"📁 模型路径: {model_path}")
    print(f"📁 路径存在: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print("❌ 模型路径不存在！")
        return False
    
    try:
        print("🔄 正在加载模型...")
        
        # 加载模型
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # 使用DPM++ 2M Karras调度器
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # 移动到GPU
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            print("✅ 模型已移动到GPU")
        
        print("✅ 模型加载成功！")
        
        # 测试推理
        print("🧪 测试推理...")
        prompt = "a beautiful landscape painting, high quality, detailed"
        
        with torch.no_grad():
            image = pipe(prompt, num_inference_steps=5).images[0]
        
        print("✅ 推理测试成功！")
        
        # 保存测试图像
        output_path = "test_output.png"
        image.save(output_path)
        print(f"💾 测试图像已保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("🎉 所有测试通过！模型工作正常！")
    else:
        print("💥 测试失败！需要检查模型或环境配置。")
