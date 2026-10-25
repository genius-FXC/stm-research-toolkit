# 环境与迁移

完整复制仓库即可保留程序与工具索引。不要复制 `.venv`，在新电脑运行 `Install.bat` 重建。通过 `Check.bat` 检查依赖。路径从文件所在目录计算，不要求固定盘符。

本地配置为 `registry/machine.json` 和 `modules/stm_line_spectrum_tool/config.json`，首次启动或安装生成，不会覆盖已有设置。移动到另一台电脑后重新配置外部软件路径。Python 安装位置可不同。

MATLAB、Scilab、Origin 不随仓库分发。历史分析模板需要自行准备上游数据与参数。Gwyddion 只用于部分独立 STP 交叉校验脚本；如需此校验，可用 `GWYDDION_HOME` 指定安装目录，入口的常规转换不要求它。

数据、输出、日志、密钥和会话不随源码发布。已有工作目录与这个发布仓库互相独立；发布整理没有迁移或修改原始数据。
