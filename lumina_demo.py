import gdsfactory as gf
from gdsfactory.generic_tech import LAYER
import numpy as np

# --- 1. 定义层 (Layer Definition) ---
# 告诉工厂：哪一层是硅，哪一层是稀土，哪一层是金属
# 格式: (GDS Layer Number, Data Type)
LAYER_WG = (1, 0)       # 硅光波导层 (Silicon Core)
LAYER_SLAB = (2, 0)     # 浅刻蚀层 (用于做纳米散射体)
LAYER_RE_PIT = (10, 0)  # 稀土生长坑位 (Rare-Earth Pit) - 我们要挖坑填材料
LAYER_METAL = (40, 0)   # 金属电极层 (Metal Contacts)

# 定义 cross_section
cross_section_wg = gf.cross_section.strip(width=0.5, layer=LAYER_WG)

@gf.cell
def lumina_pixel(
    length=20.0, 
    wg_width=0.5, 
    scatterer_size=(0.3, 0.2),
    pit_size=(0.1, 0.1) # 100nm 的坑位，用于生长 4nm 晶体聚集体
):
    """
    创建一个 LuminaCore 的标准发光像素单元
    """
    c = gf.Component()

    # A. 绘制主波导 (The Highway)
    # 这是一条 500nm 宽的硅波导
    wg = c << gf.components.straight(length=length, cross_section=cross_section_wg)
    
    # B. 绘制纳米散射体 (The Nano-Scatterer) - 基于 MEEP 仿真
    # 放置在波导上方，用于把光散射进波导
    # 我们用矩形模拟那个"光学漏斗"
    scatterer = c << gf.components.rectangle(size=scatterer_size, layer=LAYER_SLAB)
    scatterer.move((length/2 - scatterer_size[0]/2, wg_width/2 + 0.05))

    # C. 绘制稀土生长坑位 (The Emitter Pit)
    # 这是我们要在代工厂后道工序中填入稀土材料的地方
    pit = c << gf.components.rectangle(size=pit_size, layer=LAYER_RE_PIT)
    pit.move((length/2 - 0.1, wg_width/2 + 0.3))

    # D. 绘制金属电极 (Electrical Contacts)
    # 简单的示意图，展示电压是从哪里加进来的
    contact_w = 2.0
    contact = c << gf.components.rectangle(size=(contact_w, contact_w), layer=LAYER_METAL)
    contact.move((length/2 - 1.0, wg_width/2 + 1.0))
    
    # 添加端口以便后续连接
    c.add_port(name="o1", center=(0, 0), width=wg_width, orientation=180, layer=LAYER_WG)
    c.add_port(name="o2", center=(length, 0), width=wg_width, orientation=0, layer=LAYER_WG)

    return c

@gf.cell
def lumina_core_demo():
    """
    组装 RGB 阵列与路由系统
    """
    c = gf.Component("LuminaCore_v1")

    # 1. 实例化三个像素 (R, G, B)
    # 在物理版图上，它们其实结构一样，只是填充的稀土材料不同
    # 这里我们画三个串联的单元来模拟阵列
    pixel_r = c << lumina_pixel()
    pixel_g = c << lumina_pixel()
    pixel_b = c << lumina_pixel()

    # 2. 布局 (Placement)
    # 将它们排成一行 (Linear Array)
    pixel_g.connect("o1", pixel_r.ports["o2"])
    pixel_b.connect("o1", pixel_g.ports["o2"])

    # 3. 输出路由 (Output Routing)
    # 在末端连接一个光栅耦合器 (Grating Coupler)
    # 用于把计算好的光导出来，或者连接到探测器
    gc = c << gf.components.grating_coupler_elliptical(cross_section=cross_section_wg)
    gc.connect("o1", pixel_b.ports["o2"])

    # 4. 添加标签 (Labels)
    c.add_label(text="LuminaCore R-G-B Array", position=(10, 5), layer=LAYER_METAL)

    return c

# --- 主程序 ---
if __name__ == "__main__":
    # 生成顶层组件
    chip = lumina_core_demo()
    
    # 导出为 GDSII 文件 (工业标准格式)
    gds_filename = "LuminaCore_Demo_v1.gds"
    chip.write_gds(gds_filename)
    
    print(f"\n[SUCCESS] Blueprint generated: {gds_filename}")
    print("Instruction: Please open this file using 'KLayout' viewer.")
    
    # 如果你安装了 KLayout 并配置了 gdsfactory 的显示，可以直接调用：
    # chip.show()
