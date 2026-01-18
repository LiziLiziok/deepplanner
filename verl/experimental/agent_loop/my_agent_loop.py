# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
开始
  ↓
第一阶段：分解子问题 ←─────────────┐
  ↓                               │
第二阶段：检索+推理             │
  ↓                               │
检查结果                           │
  ↓                               │
是否所有子问题都成功？              │
  ├─ 是 → 进入第三阶段             │
  └─ 否 → 尝试次数 < 3？           │
            ├─ 是 → 记录失败原因 ──┘
            └─ 否 → 进入第三阶段（带失败信息）
                      ↓
第三阶段：综合答案（包含成功的子答案 + 说明失败的子问题）
  ↓
返回输出


"""
import asyncio
import json
import re
import ast
import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
import logging
logger = logging.getLogger(__name__) 

# caculate_stage = "planner"
caculate_stage = "solver"



@register("deep_planner_agent")  # 注册一个新的 agent_name
class PlannerAgentLoop(AgentLoopBase):
    """
    Planner Agent Loop: 
    1. 第一轮让模型分解问题为多个子问题
    2. 对每个子问题调用工具检索
    3. 将子问题+检索结果组合后让模型推理
    """
    
    MAX_RETRY_ATTEMPTS = 3  # 最大重试次数
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.max_model_len = self.config.actor_rollout_ref.rollout.max_model_len
    
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        
        # Initialize basic attributes
        cls.tokenizer = tokenizer
        cls.processor = processor
        
        # Initialize tools from config file
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        print(f"Initialized tools: {cls.tools}")
        
        # Load custom prompts
        planner_system_prompt_path = "verl/experimental/agent_loop/prompt/planner.md"
        retrieval_system_prompt_path = "verl/experimental/agent_loop/prompt/retrieval.md"
        solver_system_prompt_path = "verl/experimental/agent_loop/prompt/solver.md"
        with open(planner_system_prompt_path, 'r') as f:
            cls.planner_system_prompt = f.read().strip()
        with open(retrieval_system_prompt_path, 'r') as f:
            cls.retrieval_system_prompt = f.read().strip()
        with open(solver_system_prompt_path, 'r') as f:
            cls.solver_system_prompt = f.read().strip()
    
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.# <-- 单条数据！

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        messages = list(kwargs["raw_prompt"]) # 原始对话消息，单条！
        question = messages[-1]["content"]
        # 格式：[{"role": "system", "content": "..."},{"role": "user", "content": "用户问题"}] 

        request_id = uuid4().hex
        metrics = {}  # 用于 simple_timer 的字典
        
        # 记录所有尝试过程中失败的子问题（用于重试时的提示词）
        all_failed_sub_questions = []
        # 记录所有成功的子问题（用于重试时的提示词）
        all_successful_answers = []

        sub_questions = []

        plan_history=[]
        
        attempt = 0
        success = False

        all_attempts = [{"question": question}]

        while attempt < self.MAX_RETRY_ATTEMPTS and not success:
            attempt += 1
            cur_at = {
                "attempt": attempt
            }
            
            # ========== 第一阶段：Planner - 分解子问题 ==========
            planner_result = await self._planner_stage(
                messages, 
                sampling_params, 
                request_id, 
                metrics,
                plan_history=plan_history if attempt > 1 else [],
                all_successful_answers=all_successful_answers if attempt > 1 else [],
                all_failed_sub_questions=all_failed_sub_questions if attempt > 1 else [],
            )
            sub_questions = planner_result["sub_questions"]
            plan_history.append({"attempt": attempt, "sub_questions": sub_questions})
            cur_at["planner_stage"] = {
                    "sub_questions": planner_result["sub_questions"],
                    "raw_planner_response": planner_result["raw_planner_response"],
                    "user_content": planner_result["user_content"],
                    }

            # 记录最后一次 planner 阶段的输出（用于训练）
            last_planner_result = planner_result
            
            # ========== 第二阶段：对每个子问题进行检索+推理 ==========
            # 串行执行：每个子问题依赖上一个子问题的答案

            current_successful_answers = []
            current_failed_questions = []

            n = len(sub_questions)
            if not sub_questions:
                pass 
            cur_q = sub_questions[0]

            for i in range(n):
                if i+1 <= n-1:
                    next_q = sub_questions[i+1]
                else:
                    next_q = ""
                
                result = await self._retrieval_and_reasoning_stage(
                    cur_q, next_q, sampling_params, request_id, metrics, kwargs
                )

                cur_at[f"retrieval_and_reasoning_stage_{i}"] = result

                if result["success"]:
                    current_successful_answers.append({
                        "cur_sub_question": cur_q,  
                        "original_cur_sub_question": sub_questions[i], 
                        "answer": result["answer"],
                        "attempt": attempt,
                        "next_question":result["next_question"]
                    })
                    cur_q =result['next_question']
                else:
                    current_failed_questions.append({
                        "cur_sub_question": cur_q,
                        "original_cur_sub_question": sub_questions[i],
                        "answer": result["answer"],
                        "attempt": attempt,
                        "next_question":next_q
                    })
                    break

            all_attempts.append(cur_at)

            # 累积成功的答案
            all_successful_answers.extend(current_successful_answers)
            
            # 检查是否所有子问题都成功
            if not current_failed_questions:
                success = True
            else:
                # 记录失败的子问题，用于下次重试
                all_failed_sub_questions.extend(current_failed_questions)
                if attempt < self.MAX_RETRY_ATTEMPTS:
                    continue
        
        # ========== 构建第一阶段的输出 output_planner ==========
        planner_response_ids = last_planner_result["response_ids"]
        planner_response_mask = [1] * len(planner_response_ids)
        output_planner = AgentLoopOutput(
            prompt_ids=last_planner_result["prompt_ids"],
            response_ids=planner_response_ids[:self.response_length],
            response_mask=planner_response_mask[:self.response_length],
            response_logprobs=last_planner_result["log_probs"][:self.response_length] if last_planner_result["log_probs"] else None,
            num_turns=1,
            metrics=AgentLoopMetrics(
                generate_sequences=metrics.get("generate_sequences", 0.0),
                tool_calls=0.0,
            ),
            extra_fields={
                "stage": "planner",
                "sub_questions": sub_questions,
                "raw_planner_response": last_planner_result["raw_planner_response"],
                "replan_attempt_num": attempt,
            },
        )
        
        # ========== 第三阶段：综合所有子答案生成最终答案 ==========
        final_output = await self._synthesis_stage(
            messages, 
            all_successful_answers, 
            sampling_params, 
            request_id, 
            metrics, 
            kwargs,
            failed_questions=all_failed_sub_questions,
            total_attempts=attempt
        )
        output_json = {
        "user_input": final_output.extra_fields["synthesis_prompt"],
        "final_output": final_output.extra_fields["response"],
    }
        all_attempts.append({"synthesis_stage": output_json})
        
        # 返回最终输出
        if caculate_stage == "planner":
            output_planner.extra_fields["final_output_AgentLoopOutput"] = final_output
            output_planner.extra_fields["question"] = question
            output_planner.extra_fields["all_successful_sub_answers"] = all_successful_answers
            output_planner.extra_fields["all_failed_sub_questions"] = all_failed_sub_questions
            output_planner.extra_fields["all_attempts"] = all_attempts
            return output_planner
        else:
            final_output.extra_fields["output_planner_AgentLoopOutput"] = output_planner
            final_output.extra_fields["question"] = question
            final_output.extra_fields["all_successful_sub_answers"] = all_successful_answers
            final_output.extra_fields["all_failed_sub_questions"] = all_failed_sub_questions
            final_output.extra_fields["all_attempts"] = all_attempts
            
            return final_output 
    def _check_and_truncate_prompt(self, prompt_tokens: list, max_model_len: int = 10000, min_generation_len: int = 100):
        """
        检查 prompt 长度，必要时进行截断
        
        Args:
            prompt_tokens: prompt 的 token 列表
            max_model_len: 模型最大上下文长度
            min_generation_len: 保留给生成的最小 token 数
        
        Returns:
            截断后的 prompt tokens
        """
        max_prompt_len = max_model_len - min_generation_len
        
        if len(prompt_tokens) > max_prompt_len:
            logger.warning(
                f"Prompt too long ({len(prompt_tokens)} tokens), truncating to {max_prompt_len} tokens. "
                f"This may affect generation quality."
            )
            prompt_tokens = prompt_tokens[-max_prompt_len:]
        
        return prompt_tokens     
    async def _planner_stage(
        self, 
        messages: list[dict], 
        sampling_params: dict[str, Any],
        request_id: str,
        metrics: dict,
        plan_history: list[dict] = None,
        all_successful_answers: list[dict] = None,
        all_failed_sub_questions: list[dict] = None
    ) -> dict:
        """第一阶段：让模型分解问题为多个子问题
        """
        
        # 构造 Planner prompt
        system_prompt = self.planner_system_prompt
        
        system_prompt = system_prompt.replace("${last_plan}", json.dumps(plan_history, indent=2, ensure_ascii=False) if plan_history else "")
        system_prompt = system_prompt.replace("${all_successful_answers}", json.dumps(all_successful_answers, indent=2, ensure_ascii=False) if all_successful_answers else "")
        system_prompt = system_prompt.replace("${all_failed_sub_questions}", json.dumps(all_failed_sub_questions, indent=2, ensure_ascii=False) if all_failed_sub_questions else "") 

        user_question = messages[-1].get("content", "") 
        user_content = system_prompt.replace("${user_question}", user_question)

        planner_messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content}
        ]
        
        # Tokenize planner messages
        planner_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                planner_messages,
                add_generation_prompt=True,
                tokenize=True,
            ),
        )
        planner_prompt_ids = self._check_and_truncate_prompt(
            planner_prompt_ids,self.max_model_len
        )
        # 调用 LLM 生成子问题
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=planner_prompt_ids,
                sampling_params=sampling_params,
            )
        
        # 解析子问题 - 注意：output 是 TokenOutput 对象，需要用 .token_ids
        response_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        sub_questions = self._parse_sub_questions(response_text)
        
        return {
            "sub_questions": sub_questions,
            "raw_planner_response": response_text,
            "user_content": user_content,
            "prompt_ids": planner_prompt_ids,
            "response_ids": output.token_ids,
            "log_probs": output.log_probs,
        }
    
    def _resolve_sub_question(self, sub_question: str, last_step_answer: str) -> str:
        """用上一个子问题的答案替换占位符 ${last_step_answer}
        
        Args:
            sub_question: 原始子问题（可能包含占位符）
            last_step_answer: 上一个子问题的答案
            
        Returns:
            解析后的完整子问题
        """
        if "${last_step_answer}" in sub_question:
            if last_step_answer:
                return sub_question.replace("${last_step_answer}", last_step_answer)
            else:
                return sub_question.replace("${last_step_answer}", "")
        return sub_question
    
    def _parse_sub_questions(self, response_text: str) -> list[str]:
        """解析模型输出的子问题"""
        text = response_text.strip()
        # 1) 去掉 ```json 和 ``` 包裹
        text = re.sub(r"```json|```", "", text).strip()
        def extract_contents(sub_questions):
            """
            输入：list of dict 结构
            输出：仅 content 的 list，保持顺序
            """
            results = []
            for item in sub_questions:
                if isinstance(item, dict) and "content" in item:
                    results.append(item["content"])
                else:
                    raise ValueError(f"Invalid item format: {item}")
            return results        
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "sub_questions" in data:
                return extract_contents(data["sub_questions"])
        except Exception:
            pass

        try:
            data = ast.literal_eval(text)
            if isinstance(data, dict) and "sub_questions" in data:
                return extract_contents(data["sub_questions"])
        except Exception:
            pass
        return [response_text] 
    
    async def _retrieval_and_reasoning_stage(
        self,
        sub_question: str,
        next_question: str,
        sampling_params: dict[str, Any],
        request_id: str,
        metrics: dict,
        kwargs: dict
    ) -> dict:
        """第二阶段：对单个子问题进行检索+推理
        """
        
        # 2.1 构造工具调用请求
        retrieval_result = await self._call_retrieval_tool(sub_question, kwargs.get("tools_kwargs", {}))

             
        # 2.2 构造推理 prompt
        system_prompt = self.retrieval_system_prompt
        system_prompt = system_prompt.replace("${retrieval_result}", retrieval_result)
        system_prompt = system_prompt.replace("${sub_question}", sub_question)
        user_content = system_prompt.replace("${next_question}", next_question)

        reasoning_messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content}
        ]
        
        # Tokenize
        reasoning_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                reasoning_messages,
                add_generation_prompt=True,
                tokenize=True,
            ),
        )
        reasoning_prompt_ids = self._check_and_truncate_prompt(
            reasoning_prompt_ids,self.max_model_len
        )        
        # 2.3 调用 LLM 推理
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=f"{request_id}_sub_{sub_question[:10]}", 
                prompt_ids=reasoning_prompt_ids,
                sampling_params=sampling_params,
            )
        
        answer = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        raw_answer = answer

        # 检查答案是否有效
        answer = self._parse_sub_questions_answer(answer)

        if answer.get("answerable", "false").lower() != "true":
            return {
                "success": False,
                "answer": answer.get("answer", ""),
                "next_question":answer.get("next_question", ""),
                "prompt": user_content,
                "raw_answer":raw_answer
            }
        
        return {
            "success": True,
            "answer": answer.get("answer", ""),
            "next_question":answer.get("next_question", ""),
            "prompt": user_content,
            "raw_answer":raw_answer
        }
    
    def _parse_sub_questions_answer(self, response_text: str):
        """解析模型输出的子问题和答案"""
        text = response_text.strip()
        text = re.sub(r"```json|```", "", text).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                data = data[0] if data else {}
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        try:
            data = ast.literal_eval(text)
            if isinstance(data, list):
                data = data[0] if data else {}
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {
            "answerable": "false",
            "answer": response_text,
            "next_question": ""
        }
    
    async def _call_retrieval_tool(self, query: str, tools_kwargs: dict) -> str:
        """调用检索工具
        
        该方法会调用配置文件中定义的 'search' 工具进行检索。
        工具配置来自 tool_config_path（如 search_tool_config.yaml）。
        
        SearchTool 的 execute 方法接受参数:
        - query_list: list[str] - 查询字符串列表
        
        Args:
            query: 检索查询字符串
            tools_kwargs: 工具的额外参数配置
            
        Returns:
            检索结果文本，或错误信息
        """
        # 工具名称为 "search"，与 search_tool_config.yaml 中的 function.name 一致
        retrieval_tool_name = "search"
        
        if retrieval_tool_name not in self.tools:
            available_tools = list(self.tools.keys()) if self.tools else []
            return f"[检索工具未配置] 查询: {query}。可用工具: {available_tools}"
        
        tool = None
        instance_id = None
        try:
            tool = self.tools[retrieval_tool_name]
            kwargs = tools_kwargs.get(retrieval_tool_name, {})
            
            # 创建工具实例
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            
            tool_response, tool_reward, res = await tool.execute(
                instance_id, 
                {"query_list": [query]}  # 注意：参数名是 query_list，值是列表
            )
            
            return tool_response.text or ""
        except Exception as e:
            return f"[检索失败] {str(e)}"
        finally:
            if tool and instance_id:
                try:
                    await tool.release(instance_id)
                except Exception:
                    pass
    
    async def _synthesis_stage(
        self,
        original_messages: list[dict],
        sub_answers: list[dict],
        sampling_params: dict[str, Any],
        request_id: str,
        metrics: dict,
        kwargs: dict,
        failed_questions: list[dict] = None,
        total_attempts: int = 1
    ) -> AgentLoopOutput:
        """第三阶段：综合所有子答案生成最终答案
        
        Args:
            failed_questions: 最终仍然失败的子问题列表
            total_attempts: 总共尝试的次数
        """
        
        synthesis_system_prompt = self.solver_system_prompt
        sp = synthesis_system_prompt.replace("${failed_sub_questions}", json.dumps(failed_questions, indent=4))
        sp = sp.replace("${sub_qa_text}", json.dumps(sub_answers, indent=4))
        user_input = sp.replace("${question}", original_messages[-1].get("content", ""))

        synthesis_messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_input}
        ]
        
        # 直接调用LLM生成最终答案（简化第三阶段，只推理一次）
        # Tokenize synthesis messages
        synthesis_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                synthesis_messages,
                add_generation_prompt=True,
                tokenize=True,
            ),
        )
        synthesis_prompt_ids = self._check_and_truncate_prompt(
            synthesis_prompt_ids,self.max_model_len
        )        
        # 调用LLM生成最终答案
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=f"{request_id}_synthesis",
                prompt_ids=synthesis_prompt_ids,
                sampling_params=sampling_params,
            )
        
        # 获取生成的答案
        response_ids = output.token_ids
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        response_log_probs = output.log_probs
        prompt_ids = synthesis_prompt_ids
        
        # 创建response_mask（全部为1，表示所有token都是生成的）
        response_mask = [1] * len(response_ids)
        
        # 构建 AgentLoopMetrics 对象
        agent_loop_metrics = AgentLoopMetrics(
            generate_sequences=metrics.get("generate_sequences", 0.0),
            tool_calls=metrics.get("tool_calls", 0.0),
        )
        # 构造 AgentLoopOutput
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length], #前向传播计算 log_prob
            response_mask=response_mask[:self.response_length], # 决定哪些 token 参与梯度
            response_logprobs=response_log_probs[:self.response_length] if response_log_probs else None,  # 计算 PPO ratio
            num_turns=1,  # 第三阶段只进行一次推理
            metrics=agent_loop_metrics,
            extra_fields={"stage": "synthesis"},  # 传递给 reward 函数的额外信息
        )
        
        # 通过 extra_fields 属性添加额外信息
        output.extra_fields["sub_answers"] = sub_answers
        output.extra_fields["failed_questions"] = failed_questions or []
        output.extra_fields["total_attempts"] = total_attempts
        output.extra_fields["all_attempts_success"] = len(failed_questions or []) == 0
        output.extra_fields["synthesis_prompt"] = user_input
        output.extra_fields["response"] = response_text
        
        return output
