# ✨ DeepPlanner

## 🚀 快速开始

### 🛠️ 环境配置

请按照以下链接中的说明，配置**检索环境**和**训练环境**：

  * **配置指南**：
    [https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool\_examples/verl-multiturn-searchR1-like\_ZH.md](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md)

### 📊 数据集准备

#### 训练集 (待定)

目前项目**尚未进行训练**。请根据上文提到的**环境配置指南**，下载并准备 `search-r1` 训练集。

#### 🧪 评测集 (Evinote)

项目使用 `evinote` 评测集。下载后，**必须**对数据集进行以下修改以适配本项目的 Agent 脚本。

1.  **下载数据集**：根据 Evinote 项目的官方说明进行下载。
2.  **修改数据结构**：将原数据中的 `prompt` 字段替换为以下结构：
      * `role: system` 的 `content` (系统指令/`sp`) 必须**置空** (`""`)。
      * `role: user` 的 `content` (用户输入/`up`) 必须使用原数据中的 `question` 字段内容。

> ℹ️ **注意**：所有的提示词 (Prompt) 逻辑都统一在 Agent 脚本 (例如 `deepplanner_run.sh` 调用的 Python 脚本) 中调用，而不是直接写在数据集的 `system` 字段中。

**修改后的数据格式示例：**

```json
[
  {
    "content": "",
    "role": "system"
  },
  {
    "content": "swan lake the sleeping beauty and the nutcracker are three famous ballets by?",
    "role": "user"
  }
]
```

## ⚙️ 模型评测

评测前，请确保已经修改相应脚本中的**模型路径**和**评测集路径**。

### 1\. 评测 DeepPlanner

| 评测模型 | 运行脚本 | 需修改项 |
| :--- | :--- | :--- |
| DeepPlanner | `deepplanner_run.sh` | 模型路径、评测集路径 |

```bash
bash deepplanner_run.sh
```

### 2\. 评测 Search-R1 (基线模型)

| 评测模型 | 运行脚本 | 需修改项 |
| :--- | :--- | :--- |
| Search-R1 | `search_r1_run.sh` | 模型路径、评测集路径 |

```bash
bash search_r1_run.sh
```

