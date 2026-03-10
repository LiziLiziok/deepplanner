# 如何定义新 Tool 并集成到 PlannerAgentLoop

## 📋 完整流程概览

```
1. 创建 Tool 类（继承 BaseTool）
        ↓
2. 创建 YAML 配置文件
        ↓
3. 在脚本中指定配置文件路径
        ↓
4. PlannerAgentLoop 通过 self.tools["工具名"] 调用
```

---

## 步骤 1：创建 Tool 类

在 `/verl/tools/` 目录下创建你的工具文件，例如 `my_custom_tool.py`：

```python
# verl/tools/my_custom_tool.py

import logging
from typing import Any, Optional
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


class MyCustomTool(BaseTool):
    """自定义工具示例
    
    必须实现的方法：
    - __init__: 初始化工具
    - create: 创建工具实例
    - execute: 执行工具（核心逻辑）
    - calc_reward: 计算奖励（可选）
    - release: 释放资源
    """
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """初始化工具
        
        Args:
            config: 来自 YAML 配置文件的 config 部分
            tool_schema: 来自 YAML 配置文件的 tool_schema 部分
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}  # 存储每个实例的状态
        
        # 从 config 中读取你需要的配置
        self.api_url = config.get("api_url", "http://localhost:8000")
        self.timeout = config.get("timeout", 30)
        
        logger.info(f"Initialized MyCustomTool with api_url: {self.api_url}")
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """返回工具的 OpenAI 格式 schema"""
        return self.tool_schema
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """创建工具实例（每个请求创建一个实例）
        
        Returns:
            (instance_id, 初始化响应)
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        # 初始化实例状态
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
            "history": []
        }
        
        return instance_id, ToolResponse()
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """执行工具（核心方法）
        
        Args:
            instance_id: 工具实例 ID
            parameters: 模型调用时传入的参数（来自 tool_schema 中定义的参数）
        
        Returns:
            (ToolResponse, reward_score, metrics)
        """
        # 1. 从 parameters 中获取参数（名称与 tool_schema 中定义的一致）
        query = parameters.get("query", "")
        
        if not query:
            return ToolResponse(text="Error: 'query' is required"), 0.0, {}
        
        # 2. 执行你的业务逻辑
        try:
            result = await self._do_something(query)
            
            # 3. 保存结果到实例状态
            self._instance_dict[instance_id]["history"].append(result)
            
            # 4. 返回结果
            return ToolResponse(text=result), 0.0, {"status": "success"}
        
        except Exception as e:
            return ToolResponse(text=f"Error: {str(e)}"), 0.0, {"error": str(e)}
    
    async def _do_something(self, query: str) -> str:
        """你的业务逻辑"""
        # 示例：调用外部 API
        # response = requests.get(self.api_url, params={"q": query})
        # return response.text
        return f"处理了查询: {query}"
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """计算奖励（可选）"""
        return 0.0
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """释放工具实例资源"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
```

---

## 步骤 2：创建 YAML 配置文件

在 `examples/sglang_multiturn/config/tool_config/` 目录下创建 `my_custom_tool_config.yaml`：

```yaml
tools:
  - class_name: verl.tools.my_custom_tool.MyCustomTool  # 工具类的完整路径
    config:
      api_url: http://127.0.0.1:8000/api  # 你的配置参数
      timeout: 30
      type: native  # native 或 mcp
    tool_schema:
      type: function
      function:
        name: my_tool  # 工具名称（用于 self.tools["my_tool"]）
        description: 这是我的自定义工具，用于xxx功能
        parameters:
          type: object
          properties:
            query:  # 参数名称
              type: string
              description: 查询字符串
            options:  # 可选参数
              type: object
              description: 额外选项
          required:
            - query  # 必需参数列表
```

---

## 步骤 3：在启动脚本中指定配置文件

修改你的 `.sh` 脚本：

```bash
# 指定工具配置文件路径
TOOL_CONFIG="${CONFIG_DIR}/my_custom_tool_config.yaml"

# 传递给训练脚本
python ... \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    ...
```

---

## 步骤 4：在 PlannerAgentLoop 中调用

现在你可以通过 `self.tools["my_tool"]` 来调用工具了：

```python
async def _call_my_custom_tool(self, query: str, tools_kwargs: dict) -> str:
    """调用自定义工具"""
    tool_name = "my_tool"  # 与 YAML 中的 function.name 一致
    
    if tool_name not in self.tools:
        return f"[工具未配置] {tool_name}"
    
    tool = None
    instance_id = None
    try:
        tool = self.tools[tool_name]
        kwargs = tools_kwargs.get(tool_name, {})
        
        # 1. 创建实例
        instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
        
        # 2. 执行工具（参数名与 tool_schema 中定义的一致）
        tool_response, tool_reward, metrics = await tool.execute(
            instance_id,
            {"query": query}  # 传入参数
        )
        
        return tool_response.text or ""
    
    except Exception as e:
        return f"[执行失败] {str(e)}"
    
    finally:
        # 3. 释放资源
        if tool and instance_id:
            try:
                await tool.release(instance_id)
            except Exception:
                pass
```

---

## 🔑 关键点总结

| 组件 | 作用 | 位置 |
|------|------|------|
| **BaseTool** | 基类，定义接口 | `verl/tools/base_tool.py` |
| **你的 Tool 类** | 实现具体逻辑 | `verl/tools/my_custom_tool.py` |
| **YAML 配置** | 定义类路径、参数、schema | `config/tool_config/xxx.yaml` |
| **initialize_tools_from_config** | 加载 YAML 并实例化工具 | 自动调用 |
| **self.tools** | 工具字典（类属性） | 继承自 ToolAgentLoop |

---

## 📊 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                    启动时初始化                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  YAML 配置文件                                               │
│       │                                                     │
│       ▼                                                     │
│  initialize_tools_from_config()                             │
│       │                                                     │
│       ├──► 读取 class_name: verl.tools.my_tool.MyTool       │
│       │                                                     │
│       ├──► 动态导入类: get_tool_class(class_name)           │
│       │                                                     │
│       └──► 实例化: MyTool(config, tool_schema)              │
│                   │                                         │
│                   ▼                                         │
│       cls.tools = {"my_tool": <MyTool instance>}            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    运行时调用                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PlannerAgentLoop._call_my_custom_tool(query)               │
│       │                                                     │
│       ├──► tool = self.tools["my_tool"]                     │
│       │                                                     │
│       ├──► instance_id, _ = await tool.create()             │
│       │                                                     │
│       ├──► response, _, _ = await tool.execute(             │
│       │        instance_id, {"query": query})               │
│       │                                                     │
│       └──► await tool.release(instance_id)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

需要我帮你创建一个具体的自定义工具示例吗？比如一个简单的 HTTP API 调用工具？



# 这个脚本使用的检索工具
从脚本中可以看到，工具配置文件路径是：

```bash
TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"
```

让我读取这个配置文件来了解检索工具的详细信息。
[读取文件]
[查找文件]
让我进一步查看SearchTool的具体实现，了解其输入输出格式：
[读取文件]
让我再查看一下实际的搜索执行函数 `perform_single_search_batch` 的实现，以了解输出格式：
[读取文件]
现在我已经获得了完整的信息。让我为你总结这个脚本的检索工具的详细信息：

## 检索工具总结

### 1. 工具名称
**`search`** - 一个基于 Web 检索的搜索工具

### 2. 工具配置
配置文件位于：[search_tool_config.yaml](/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/verl/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml)

```yaml
tools:
  - class_name: verl.tools.search_tool.SearchTool
    config:
      retrieval_service_url: http://127.0.0.1:8000/retrieve
      num_workers: 120
      rate_limit: 120
      timeout: 30
      type: native
```

### 3. 调用格式（OpenAI Function Calling 格式）

```json
{
  "type": "function",
  "function": {
    "name": "search",
    "description": "Searches the web for relevant information based on the given query.",
    "parameters": {
      "type": "object",
      "properties": {
        "query_list": {
          "type": "array",
          "item": {
            "type": "string"
          },
          "description": "A list of fully-formed semantic queries. The tool will return search results for each query."
        }
      },
      "required": ["query_list"]
    }
  }
}
```

### 4. 输入格式

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `query_list` | `array[string]` | ✅ | 语义查询列表，每个元素是一个完整的查询字符串 |

**输入示例：**
```json
{
  "query_list": ["What is machine learning?", "How does neural network work?"]
}
```

### 5. 输出格式

输出是一个 JSON 字符串，包含 `result` 字段：

**成功时的输出格式：**
```json
{
  "result": "Doc 1 (Title: 标题1)\n正文内容1\n\nDoc 2 (Title: 标题2)\n正文内容2\n\n---\nDoc 1 (Title: 标题3)\n正文内容3\n\n..."
}
```

- 每个查询的结果用 `---` 分隔
- 每个文档格式为：`Doc N (Title: 标题)\n正文内容`
- 默认返回 `topk=3` 个结果

**错误时的输出格式：**
```json
{
  "result": "Search error: 错误信息"
}
```

或
```json
{
  "result": "No search results found."
}
```

### 6. 工作流程

```
┌─────────────────┐
│   模型生成调用    │
│ search(query_list) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SearchTool    │
│  execute()方法   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Ray执行池       │
│ (并发控制+限流)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HTTP POST 请求  │
│ 127.0.0.1:8000  │
│   /retrieve     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  格式化结果返回   │
└─────────────────┘
```

### 7. 关键特性

- **并发控制**：通过 Ray Actor 实现 Token Bucket 限流（默认 120 并发）
- **重试机制**：最多重试 10 次，延迟递增
- **超时设置**：默认 30 秒
- **结果格式化**：自动将检索结果转换为可读的文档格式
