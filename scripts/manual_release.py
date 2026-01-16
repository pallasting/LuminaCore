#!/usr/bin/env python3
"""
手动发布脚本 - 用于在没有 PyPI token 的情况下上传
使用 GitHub Release 下载和 PyPI 上传
"""

import os
import sys
import subprocess
import webbrowser

def run_command(cmd, check=True):
    """运行命令并处理错误"""
    print(f"执行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False, result.stderr
    return True, result.stdout

def main():
    print("🚀 LuminaCore v0.2.0 手动发布指南")
    print("="*50)
    
    # 1. 检查文件
    print("\n📦 检查构建产物...")
    if not os.path.exists("dist"):
        print("❌ dist 目录不存在")
        return False
    
    files = os.listdir("dist")
    print(f"✅ 找到文件: {files}")
    
    # 2. 提供手动上传链接
    print("\n🔗 手动上传链接:")
    
    # GitHub Release
    github_url = "https://github.com/pallasting/LuminaCore/releases/new"
    print(f"\n📝 创建 GitHub Release:")
    print(f"   访问: {github_url}")
    print(f"   标题: LuminaFlow v0.2.0: 集成 Rust 后端高性能光学计算内核")
    print(f"   标签: v0.2.0")
    print(f"   描述文件: RELEASE_v0.2.0.md")
    print(f"   附加文件:")
    print(f"     - dist/lumina_flow-0.2.0.tar.gz")
    print(f"     - dist/lumina_flow-0.2.0-py3-none-any.whl")
    
    # PyPI 上传
    pypi_url = "https://pypi.org/account/login/"
    print(f"\n📤 上传到 PyPI:")
    print(f"   访问: {pypi_url}")
    print(f"   选择 'Upload file'")
    print(f"   上传文件:")
    print(f"     - lumina_flow-0.2.0.tar.gz")
    print(f"     - lumina_flow-0.2.0-py3-none-any.whl")
    
    # 3. 测试安装
    print(f"\n🧪 测试安装:")
    print(f"   pip install lumina-flow==0.2.0")
    print(f"   python -c 'import lumina; print(\"✅ 成功!\")'")
    
    # 4. 自动打开浏览器
    print(f"\n🌐 正在打开发布页面...")
    try:
        webbrowser.open(github_url)
        print(f"✅ 已打开 GitHub Release 页面")
    except:
        print(f"❌ 无法自动打开浏览器，请手动访问: {github_url}")
    
    print(f"\n🎉 发布准备完成！")
    print(f"📋 完成后请测试:")
    print(f"   1. pip install lumina-flow==0.2.0")
    print(f"   2. python -c 'import lumina; from lumina.layers import OpticalLinear; print(\"✅ 导入成功!\")'")
    print(f"   3. 测试 Rust 后端: export LUMINA_USE_RUST=1")
    
    return True

if __name__ == "__main__":
    main()