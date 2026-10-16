# 任务示例：SINDy 动力系统辨识

本节展示一份 Harbor 科学计算任务的输入、程序接口和输出约定。该示例使用合成观测数据，考察从数据恢复动力学方程的能力。

## 任务目标

给定参数化 Hopf 系统的带噪状态与导数观测，恢复两条方程在指定单项式库上的系数，并预测查询参数下的增长率、稳定振幅和分岔位置：

```text
xdot = (b0 + b1*mu)*x - omega*y - a*(x² + y²)*x
ydot = omega*x + (b0 + b1*mu)*y - a*(x² + y²)*y
```

其中 `b0`、`b1`、`a`、`omega` 未知，且 `a > 0`、`b1 > 0`、`omega > 0`。候选基函数为三个变量 `x`、`y`、`mu` 的总次数不超过 3 的单项式，共 20 项。

## 输入文件

| 文件 | 内容 |
| --- | --- |
| `samples.csv` | 900 行 `x,y,mu,xdot,ydot` 观测，导数含小幅高斯噪声 |
| `library.json` | 20 个单项式的指数及列顺序 |
| `queries.json` | 4 个待预测的 `mu` 值 |
| `README.md` | 方程符号、文件格式和输出说明 |

观测数据片段：

```csv
x,y,mu,xdot,ydot
0.25155190122069637,1.2631956772361304,-0.80225488734348971,-2.7852517274645363,-2.3684885798716087
-0.049753970454211593,0.037702180145090303,-0.80225488734348971,-0.023963676199332287,-0.11981683825047357
```

查询文件使用以下结构：

```json
{"query_mu": [-0.5496858751490484, -0.18767029100368254, 0.2080211614342755, 0.5447798443601972]}
```

## 程序接口

做题者编写 `/app/solver.py`，读取指定目录中的输入，并将结果写入输出目录：

```bash
python /app/solver.py --case-dir /app/public_case --output /app/output
```

输出文件为 `/app/output/solution.json`：

| 字段 | 类型与要求 |
| --- | --- |
| `coefficients` | 2×20 数值矩阵；两行分别对应 `xdot`、`ydot`，列顺序与 `library.json` 一致 |
| `query_mu` | 查询参数数组，顺序与输入一致 |
| `sigma` | 对应的增长率数组，`sigma = b0 + b1*mu` |
| `amplitude` | 对应的稳定振幅数组；`sigma > 0` 时为 `sqrt(sigma/a)`，其余为 0 |
| `mu_c` | 分岔位置，`-b0/b1` |

增长率、振幅和分岔位置应从恢复的系数计算。支持一般最小二乘、阈值回归等数值方法，评分不限定求解算法。

## 任务包结构

```text
hopf-sindy-identification/
├── instruction.md
├── task.toml
├── environment/
│   ├── Dockerfile
│   ├── starter/
│   └── data/
├── generation_data.py
├── generation/Dockerfile
├── solution/
└── tests/
    ├── Dockerfile
    ├── test_outputs.py
    ├── support/
    └── data/
```

`environment/data/` 保存公开输入，`solution/` 保存参考解，`tests/data/` 保存隐藏案例和评分真值。参考解和隐藏真值不进入做题镜像。起始程序提供待实现的调用接口。

运行环境使用 Python 3.12、NumPy 1.26.4、2 CPU 和 2048 MB 内存。数据由任务生成器合成，不依赖外部数据下载。

## 评分约定

三个隐藏案例使用不同的系数和随机种子。评分器比较全部系数、增长率、振幅和分岔位置：

- 系数允许误差为 `0.01 + 0.02 × |真值|`。
- 派生量允许误差为 `0.01 + 0.03 × |真值|`。
- 所有隐藏案例及全部字段通过才得 1 分，否则为 0 分。

生产阶段检查任务格式、数据生成的确定性、参考解通过及空解失败。此类执行检查不提供任务难度标定。

任务生产接口见 [Graph 使用说明](graph-task-producer.md)，公共输入与输出类型见[任务生产协议](task-protocols.md)。
