# 发布验证（2026-10-25）

在独立发布目录新建虚拟环境并安装 requirements.txt，未修改原运行环境。测试平台：Windows AMD64、CPython 3.14.3。

实际依赖：NumPy 2.5.4、SciPy 1.18.1、Matplotlib 3.11.2、Pillow 12.3.0、PySide6 6.12.0、pyqtgraph 0.14.0、PyOpenGL 3.1.10。README 所列 Python 3.11+ 是安装下限，本次没有逐一测试所有 Python/Windows 版本。

- 安装：干净环境依赖安装成功；setup.ps1 重复运行成功；可选 MATLAB/Scilab/Origin 与核心依赖分开报告。未在无 Python 的电脑实测 winget 安装分支。
- 入口：15 个注册工具的入口文件存在，Qt 入口窗口构造和渲染成功。README 截图来自无实验数据的发布版。
- 窗口初始化：2D-lockin 与 Band Analysis 构造成功；线谱服务以空数据集初始化成功。未逐个点击验证所有 GUI 功能。
- 2D-lockin：原有 unittest 共 21 项，20 项通过，1 项真实 SXM 测试因未提供实验文件跳过；AI 用例只测试本地构造/解析，不调用服务。
- 线谱：合成数据离散采样与 profile FFT 检查通过；依赖旧实验文件的部分未运行。
- 批处理：合成 3DS 导出 6 张 SXM，形貌数值逐点一致；生成 6 个 FFT/STP；iFFT 生成 STP/NPZ/CSV，输出有限且输入哈希不变。
- iFFT 相位保留/波矢通过与抑制检查最大误差约 3.28e-15；拒绝覆盖非空输出；会话保存与加载通过。

本次没有重新进行真实材料分析、MATLAB Kondo 拟合、Scilab Ternes 拟合或 Origin 项目导出验证。代码发布验证不意味着外部软件在所有版本中兼容，也不意味着模板适用于未检查的新数据。
