# 数据集分析模板

`modules/stm_line_spectrum_tool/research_quicklooks` 保留先前分析的程序和相互依赖关系，不包含对应实验数据。目录日期只是流程版本标识。

这些模板可能在导入时读取上游结果并创建输出目录，不应当作通用库导入。先读完整脚本，按自己的数据检查尺寸、行号、偏压区间、峰定义、平滑窗口和坐标约定；保留原有校验，不能为了让程序运行而删除断言。

已把个人电脑绝对路径改为显式环境变量。只设置需要运行的脚本所使用的变量：

| 环境变量 | 输入 |
| --- | --- |
| `STM_DATA_FILE` | 使用者自己的 Nanonis 3DS |
| `STM_REGION_MASKS` | 区域 mask 的 NPZ |
| `STM_REGION_DEFINITIONS` | 区域与坐标定义 JSON |
| `STM_SITE_COORDINATES` | 原子位点坐标 CSV |
| `STM_QUALITY_REPORT` | 数据质量与坐标检查 JSON |
| `STM_REFERENCE_SCRIPT` | 可选兼容性测试使用的参考处理程序 |
| `STM_SXM_INPUT` | 历史独立 FFT/STP 脚本输入目录；入口批处理使用任务参数 |

示例：`$env:STM_DATA_FILE = (Resolve-Path .\data\my-grid.3ds).Path`。

峰位/能隙流程仍依赖上游 `Se_rows_frequency_energy.npz` 等文件及相应元数据；二阶导数与周期性流程也有上游结果要求。设置一个 3DS 路径不会自动补齐全部输入。建议通过入口复制工具上下文给分析助手，再共同确定区域和模型。

公开版保留 Ternes 核心和通用 `ai_fit_two_spin.sce` 入口，不包含历史针对特定实验的批处理任务、已拟合参数及结果报告。脚本名称中的 `ai` 不代表执行需要 AI 服务：其数据路径、输出和模型参数由调用者配置。
