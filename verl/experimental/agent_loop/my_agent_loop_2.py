# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""


"""
import asyncio
import json
import copy
import re
import ast
import time
from typing import Any
from uuid import uuid4
from enum import Enum
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.profiler import simple_timer
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from datetime import datetime

import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


MAX_SEARCH_TURNS = 3

target_sequences = ["</plan>", " </plan>", "</plan>\n", " </plan>\n", "</plan>\n\n", " </plan>\n\n"]

DEBUG_LOG_DIR = "/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/verl/logs/deep_planner_agent_2-0208-loop-log"
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)

class AgentState(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    SOLVING = "solving"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        # State variables
        self.prompt_ids: list[int] = []#是planner
        self.response_ids: list[int] = []
        self.response_text: str = ""
        self.search_query: str = ""
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.search_turns : int = 0
        self.final_answer: str = ""
        self.debug_log_data: dict = {}

        # planner
        self.sp_planner: str = ""
        self.question: str = ""
        self.history_results_cur: list[dict[str, Any]] = []
        self.history_results_all: list[dict[str, Any]] = [] # {"sub_question": str, "answer": str},储存所有
        self.history_results_success_all: list[dict[str, Any]] = []
        self.history_results_failed_all: list[dict[str, Any]] = []
        self.planner_sampling_params: dict[str, Any] = {}

        # Retrieval
        self.sp_retriever: str = ""
        self.retrieval_query: list[dict[str, Any]] = [] # 存储当前轮所有检索query
        self.retrieval_query_cur: str = "" 
        self.retrieval_result_cur: str = "" 
        self.retriever_sampling_params: dict[str, Any] = {}

        # solver
        self.sp_solver: str = ""
        self.solver_result: str = ""
        self.solver_sampling_params: dict[str, Any] = {}
     
        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Extra fields for dynamic addition
        self.extra_fields: dict[str, Any] = {}


@register("deep_planner_agent_2")  # 注册一个新的 agent_name
class PlannerAgentLoop_2(AgentLoopBase):
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
        cls.max_search_turns = MAX_SEARCH_TURNS
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        print(f"*****************************Initialized tools: {cls.tools}")
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        print(f"Initialized tools: {cls.tools}")

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        
        # Initialize tools from config file
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        print(f"Initialized tools: {cls.tools}")
        
        # Load custom prompts
        planner_system_prompt_path = "/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/verl/verl/experimental/agent_loop/prompt/0203/planner_en.md"
        retrieval_system_prompt_path = "/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/verl/verl/experimental/agent_loop/prompt/0203/retrieval_en.md"
        solver_system_prompt_path = "/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/verl/verl/experimental/agent_loop/prompt/0203/solver.md"
        with open(planner_system_prompt_path, 'r') as f:
            cls.planner_system_prompt = f.read().strip()
        with open(retrieval_system_prompt_path, 'r') as f:
            cls.retrieval_system_prompt = f.read().strip()
        with open(solver_system_prompt_path, 'r') as f:
            cls.solver_system_prompt = f.read().strip()

    async def _save_debug_log(self, log_data: dict) -> None:
        """
        Save detailed debug log to jsonl file for tracing the processing flow.
        
        Args:
            log_data: Dictionary containing all debug information
        """
        try:
            # Ensure log directory exists
            os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
            
            # Generate log file path with date
            date_str = datetime.now().strftime("%Y%m%d")
            log_file_path = os.path.join(DEBUG_LOG_DIR, f"debug_trace_{date_str}.jsonl")
            
            # Write to jsonl file (append mode)
            def write_log():
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            
            await self.loop.run_in_executor(None, write_log)
            
        except Exception as e:
            logger.error(f"Failed to save debug log: {e}")   

    async def _save_debug_log_start(self, log_data: dict) -> None:
        """
        Save detailed debug log to jsonl file for tracing the processing flow.
        
        Args:
            log_data: Dictionary containing all debug information
        """
        try:
            # Ensure log directory exists
            os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
            
            # Generate log file path with date
            date_str = datetime.now().strftime("%Y%m%d")
            log_file_path = os.path.join(DEBUG_LOG_DIR, f"debug_trace_debug.jsonl")
            
            # Write to jsonl file (append mode)
            def write_log():
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            
            await self.loop.run_in_executor(None, write_log)
            
        except Exception as e:
            logger.error(f"Failed to save debug log: {e}")   

  
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.# <-- 单条数据！

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        print("===============================Starting agent loop==============================")
        messages = list(kwargs["raw_prompt"])
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        sampling_params = copy.deepcopy(sampling_params)
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        # Use string stop sequences - requires skip_tokenizer_init=False in rollout config
        sampling_params["stop"] = target_sequences
        sampling_params['logprobs'] = True

        await self._save_debug_log_start({"request_id": request_id,"messages": messages,"sampling_params": sampling_params})

        agent_data = AgentData(
            messages=messages,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            image_data=image_data,
        )

        agent_data.question = kwargs.get("question", "")

        agent_data.sp_planner = self.planner_system_prompt
        agent_data.sp_retriever = self.retrieval_system_prompt
        agent_data.sp_solver = self.solver_system_prompt

        agent_data.planner_sampling_params =  {
                            "temperature": 1,
                            "top_p": 1,
                            "repetition_penalty": 1,
                            "logprobs": True,
                            "stop": target_sequences
}
        agent_data.retriever_sampling_params = {
    "temperature": 1,
    "top_p": 1,
    "repetition_penalty": 1,
  }
        agent_data.solver_sampling_params = {
    "temperature": 0,
    "top_p": 1,
    "repetition_penalty": 1,
  }
        

        agent_data.debug_log_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "input_text": "",
            "response_length_limit": self.response_length,
            "max_search_turns": self.max_search_turns,
            "turns": [],  # Each turn's detailed info
        } 

        state = AgentState.PENDING
        state = await self._handle_pending_state(agent_data)
        initial_prompt_length = len(agent_data.prompt_ids)

        while state != AgentState.TERMINATED:
            if state == AgentState.PLANNING:
                state = await self._handle_planning_state(agent_data)
            elif state == AgentState.SOLVING:
                state = await self._handle_solving_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED  

        # Finalize output
        input_ids = agent_data.prompt_ids[:initial_prompt_length]
        response_ids = agent_data.prompt_ids[initial_prompt_length:]

        final_output = AgentLoopOutput(
            prompt_ids=input_ids,
            response_ids=response_ids,
            response_mask=agent_data.response_mask,
            response_logprobs=agent_data.response_logprobs
            if agent_data.response_logprobs
            else None,
            num_turns= agent_data.search_turns ,
            metrics=agent_data.metrics,
            extra_fields={
                "final_answer": agent_data.final_answer,
                "search_turns": agent_data.search_turns,
                "question": agent_data.messages[-1]["content"],
                "response_text": agent_data.response_text,
            },
        )
        agent_data.debug_log_data["question"] = agent_data.messages[-1]["content"]
        agent_data.debug_log_data["history_results_cur"] = agent_data.history_results_cur
        agent_data.debug_log_data["history_results_all"] = agent_data.history_results_all
        agent_data.debug_log_data["history_results_success_all"] = agent_data.history_results_success_all
        agent_data.debug_log_data["history_results_failed_all"] = agent_data.history_results_failed_all

        agent_data.debug_log_data["final_answer"] = agent_data.final_answer
        agent_data.debug_log_data["search_turns"] = agent_data.search_turns
        agent_data.debug_log_data["response_text"] = agent_data.response_text
        agent_data.debug_log_data["ids"] = {
                "prompt_ids_len":len(input_ids),"response_ids_len":len(response_ids),"response_mask_len":len(agent_data.response_mask),"response_logprobs_len":len(agent_data.response_logprobs),
                "prompt_ids": input_ids, "response_ids": response_ids,"response_mask": agent_data.response_mask,"response_logprobs": agent_data.response_logprobs
                }
        agent_data.debug_log_data["metrics"] = agent_data.metrics

        await self._save_debug_log(agent_data.debug_log_data)
        return final_output   

    def _construct_planner_prompt(self, agent_data: AgentData) -> list:
        """Construct the planner prompt from agent data."""
        prompt = agent_data.sp_planner+agent_data.question
        messages = [{"role": "system", "content": "You are a helpful and harmless assistant."},{"role": "user", "content": prompt}]
        return messages  
    def _construct_retrieval_prompt(self, agent_data: AgentData) -> list:
        """Construct the retrieval prompt from agent data."""
        prompt = agent_data.sp_retriever.replace("$Question", agent_data.retrieval_query_cur)
        prompt = prompt.replace("$Retrieved Documents", agent_data.retrieval_result_cur)
        messages = [{"role": "system", "content": "You are a helpful and harmless assistant."},{"role": "user", "content": prompt}]
        return messages  
    def _construct_solver_prompt(self, agent_data: AgentData) -> list:
        """Construct the solver prompt from agent data."""
        qa = ""
        for item in agent_data.history_results_success_all:
            qa += f"Q: {item['sub_question']} A: {item['answer']}\n"
        prompt = agent_data.sp_solver.replace("$Question", agent_data.question).replace("$sub_qa_text", qa)
        messages = [{"role": "system", "content": "You are a helpful and harmless assistant."},{"role": "user", "content": prompt}]
        return messages  
       
    async def _handle_pending_state(self, agent_data: AgentData) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        agent_data.messages = self._construct_planner_prompt(agent_data)
        agent_data.prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                agent_data.messages,
                add_generation_prompt=True,
                tokenize=True,
            ),
        )

        text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(agent_data.prompt_ids, skip_special_tokens=False)
        )
        agent_data.debug_log_data["messages"] = agent_data.messages
        agent_data.debug_log_data["input_text"] = text 

        return AgentState.PLANNING

    async def _handle_retrieval_pending_state(self, agent_data: AgentData) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        messages = self._construct_retrieval_prompt(agent_data)
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            ),
        )

        text_input = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        )
        return prompt_ids,text_input
    
    async def _handle_solving_pending_state(self, agent_data: AgentData) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        messages = self._construct_solver_prompt(agent_data)
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            ),
        )

        text_input = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        )
        return prompt_ids,text_input
    
    def _parse_llm_response_content(self, content: str) -> tuple[str, list[dict[str, Any]]]:
        """Parse the LLM response into a message and tool calls."""
        try:
            print(f"-------*****planning llm response:  {content}")
            content = ast.literal_eval(content)
            if not isinstance(content, list):
                content = None
        except Exception as e:
            content = None
        return content  
    
    def _parse_llm_response(self, response: str) -> tuple[str, list[dict[str, Any]]]:
        """Parse the LLM response into a message and tool calls."""
        pattern = r'<(plan)>(.*?)</\1>'
        match = re.search(pattern, response, re.DOTALL)
        reason = ""
        if match:
            content = match.group(2).strip()  # Return only the content inside the tags
            action = match.group(1)
            content_ = self._parse_llm_response_content(content)
            if content_ is not None:
                content = content_
            else:
                reason = f"The output could not be parsed. Please regenerate a Python-parsable list where each item contains an integer 'id' and a string 'content'."

        else:   
            content = ''
            action = None
            reason = f"Please follow the format and generate the planner as: <plan>…</plan>"
        return content, action, reason


    async def _handle_planning_state(
        self, agent_data: AgentData
    ) -> AgentState:
        """Handle the planning state: generate model response and check for tool calls."""
    
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=agent_data.planner_sampling_params,
                image_data=agent_data.image_data,
            )
       
        llm_input_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(agent_data.prompt_ids, skip_special_tokens=False)
        )
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids

        llm_response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
        )
        agent_data.response_text+= llm_response_text
        agent_data.response_mask += [1] * len(agent_data.response_ids)

        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        else:
            agent_data.response_logprobs += [-99] * len(agent_data.response_ids)

        agent_data.debug_log_data["turns"].append({
            "turn": f"{agent_data.search_turns}_planner",
            "llm_input_text": llm_input_text,
            "llm_response_text": llm_response_text,
            "ids":{
                "prompt_ids_len":len(agent_data.prompt_ids),"response_ids_len":len(agent_data.response_ids),"response_mask_len":len(agent_data.response_mask),"response_logprobs_len":len(agent_data.response_logprobs),
                "prompt_ids": agent_data.prompt_ids, "response_ids": agent_data.response_ids,"response_mask": agent_data.response_mask,"response_logprobs": agent_data.response_logprobs
                }
        })

        content, action, reason = self._parse_llm_response(llm_response_text)
        if reason:
            agent_data.debug_log_data["turns"][-1]["trigger"] = "error"
            agent_data.debug_log_data["turns"][-1]["trigger_content"] = ""
            text_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(reason, add_special_tokens=False)
            )
            agent_data.response_ids = text_ids
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_text += reason
            agent_data.response_mask += [0] * len(text_ids)
            agent_data.response_logprobs += [0] * len(text_ids)
            agent_data.debug_log_data["turns"][-1]["trigger"] = "invalid_action"
            agent_data.debug_log_data["turns"][-1]["trigger_content"] = reason
            agent_data.debug_log_data["turns"][-1]["ids"] = {
                "prompt_ids_len":len(agent_data.prompt_ids),"response_ids_len":len(agent_data.response_ids),"response_mask_len":len(agent_data.response_mask),"response_logprobs_len":len(agent_data.response_logprobs),
                "prompt_ids": agent_data.prompt_ids, "response_ids": agent_data.response_ids,"response_mask": agent_data.response_mask,"response_logprobs": agent_data.response_logprobs
                }
            return AgentState.PLANNING
        else:
            agent_data.debug_log_data["turns"][-1]["trigger"] = action
            agent_data.debug_log_data["turns"][-1]["trigger_content"] = content
            agent_data.retrieval_query = content

            return AgentState.PROCESSING_TOOLS
    def _process_retrieval_query(self,query_list,id_,answer_list):
        cur_q = query_list[id_]["content"]
        matches = re.findall(r"\{#(\d+)\}", cur_q)
        if matches:
            for match in matches:
                cur_q = cur_q.replace(f"{{#{match}}}", answer_list[int(match)]["answer"])
        return cur_q    

    async def _retrieval_tool(self,agent_data: AgentData,retrieval_tool_name):

        tool = None
        instance_id = None
        try:
            tool = self.tools[retrieval_tool_name]
            kwargs = agent_data.tools_kwargs.get(retrieval_tool_name, {})
            
            # Create tool instance
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            
            # Execute search - SearchTool.execute expects query_list parameter
            tool_response, tool_reward, res = await tool.execute(
                instance_id,
                {"query_list": [agent_data.retrieval_query_cur]}
            )
            text = tool_response.text
            text = json.loads(text)["result"] 
        except Exception as e:
            text = f"Error executing retrieval tool: {str(e)}"
            logger.error(f"Error executing retrieval tool: {str(e)}")
        finally:
            if tool and instance_id:
                try:
                    await tool.release(instance_id)
                except Exception:
                    pass
        return text


    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        """
        Call the search/retrieval tool.
        
        Args:
            query: Search query string
            tools_kwargs: Tool configuration kwargs
            
        Returns:
            Retrieved text content
        """
        retrieval_tool_name = "search"
        
        if retrieval_tool_name not in self.tools:
            available_tools = list(self.tools.keys()) if self.tools else []
            error_text = f"[Retrieval tool not configured] Query: . Available tools: {available_tools}\n"
            error_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(error_text, add_special_tokens=False)
            )
            return AgentState.TERMINATED
        
        agent_data.debug_log_data["turns"].append({
            "turn": f"{agent_data.search_turns}_retrieval",
            })        
        for item in agent_data.retrieval_query:
            id = item['id']
            agent_data.debug_log_data["turns"][-1][f"id_{id}"] = {
                "query": item['content'],
            }


            query = self._process_retrieval_query(agent_data.retrieval_query,id,agent_data.history_results_cur)
            agent_data.retrieval_query_cur = query
            agent_data.retrieval_result_cur = await self._retrieval_tool(agent_data,retrieval_tool_name)
            prompt_ids,text_input = await self._handle_retrieval_pending_state(agent_data)
            agent_data.debug_log_data["turns"][-1][f"id_{id}"]["llm_input"] = text_input

            cnt=0
            while cnt < 3:
                with simple_timer("retrieval_generate", agent_data.metrics):
                    output = await self.server_manager.generate(
                        request_id=agent_data.request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=agent_data.retriever_sampling_params,
                        image_data=agent_data.image_data,
                    )
                response_ids = output.token_ids

                llm_response_text = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(response_ids, skip_special_tokens=False)
                )
                agent_data.debug_log_data["turns"][-1][f"id_{id}"]["llm_output"] = llm_response_text
                try:
                    llm_response_text = json.loads(llm_response_text)
                    break
                except Exception as e:
                    logger.error(f"Error parsing llm_response_text: {e}")
                    cnt += 1
                    if cnt == 3:
                        return AgentState.TERMINATED
                    continue
            
            res = {
                "id": id,
                "sub_question": item['content'],
                "answer": llm_response_text["answer"],
                "answerable": llm_response_text["answerable"]
            }
            agent_data.history_results_cur.append(res)
            agent_data.history_results_all.append(res)

            if llm_response_text["answerable"] == "false":
                agent_data.history_results_failed_all.append(res)
                agent_data.debug_log_data["turns"][-1][f"id_{id}"]['answerable'] = "false"
                agent_data.debug_log_data["turns"][-1][f"id_{id}"]['answer'] = llm_response_text["answer"]
                agent_data.search_turns += 1
                if agent_data.search_turns <= self.max_search_turns:
                    text = ""
                    for i in agent_data.history_results_cur:
                        if i['answerable'] == "false":
                            
                            text += json.dumps({
                                "sub_question": i['sub_question'],
                                "answer": "failed to find answer," + i['answer'],
                            }, ensure_ascii=False) + "\n"
                        else:
                            text += json.dumps({
                                "sub_question": i['sub_question'],
                                "answer": i['answer'],
                            }, ensure_ascii=False) + "\n"
                    text = "<retrieval>\n" + text + "</retrieval>\n"
                    text_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.encode(text, add_special_tokens=False)   
                    )
                    agent_data.prompt_ids+= text_ids
                    agent_data.response_mask += [0] * len(text_ids)
                    agent_data.response_logprobs += [0] * len(text_ids)
                    agent_data.history_results_cur = []
                    return AgentState.PLANNING
                else:
                    return AgentState.TERMINATED
            
            else:
                agent_data.history_results_success_all.append(res)
                agent_data.debug_log_data["turns"][-1][f"id_{id}"]['answerable'] = "true"
                agent_data.debug_log_data["turns"][-1][f"id_{id}"]['answer'] = llm_response_text["answer"]
        agent_data.search_turns += 1    
        return AgentState.SOLVING

    async def _handle_solving_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        """
        Call the search/retrieval tool.
        
        Args:
            query: Search query string
            tools_kwargs: Tool configuration kwargs
            
        Returns:
            Retrieved text content
        """
        agent_data.debug_log_data["turns"].append({
            "turn": f"{agent_data.search_turns}_solve",
            })        


        prompt_ids,text_input = await self._handle_solving_pending_state(agent_data)
        agent_data.debug_log_data["turns"][-1]["llm_input"] = text_input

        with simple_timer("retrieval_generate", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=prompt_ids,
                sampling_params=agent_data.solver_sampling_params,
                image_data=agent_data.image_data,
            )
        response_ids = output.token_ids
        llm_response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(response_ids, skip_special_tokens=False)
        )

        agent_data.debug_log_data["turns"][-1]["llm_output"] = llm_response_text
        agent_data.final_answer = llm_response_text
        return AgentState.TERMINATED

