# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4
import re
from datetime import datetime

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


MAX_SEARCH_TURNS = 3

target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]

DEBUG_LOG_DIR = "/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/verl/logs/search-r1-agent-0201-loop-log"
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)

class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        image_data: Any = None,  # Add image_data parameter
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_text: str = ""
        self.search_query: str = ""
        self.search_query_list: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.search_turns : int = 0
        self.final_answer: str = ""
        self.debug_log_data: dict = {}
     


        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Extra fields for dynamic addition
        self.extra_fields: dict[str, Any] = {}


@register("search_r1_agent_loop_0201")
class SearchR1AgentLoop_0201(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
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
        # Note: When using string-based stop sequences, make sure skip_tokenizer_init=False in rollout config

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

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
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

        agent_data.debug_log_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "input_text": "",
            "response_length_limit": self.response_length,
            "max_search_turns": self.max_search_turns,
            "turns": [],  # Each turn's detailed info
        }   


        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            # if agent_data.search_turns >= self.max_search_turns:
            #     break
            if len(agent_data.prompt_ids) >= 25000:
                break
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
                initial_prompt_length = len(agent_data.prompt_ids)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
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
        agent_data.debug_log_data["search_query_list"] = agent_data.search_query_list
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

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
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
        return AgentState.GENERATING

    def _parse_llm_response(self, response: str) -> tuple[str, list[dict[str, Any]]]:
        """Parse the LLM response into a message and tool calls."""
        pattern = r'<(search|answer)>(.*?)</\1>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            content = match.group(2).strip()  # Return only the content inside the tags
            action = match.group(1)
        else:
            content = ''
            action = None
        return content, action

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )
        print("---------output:", output)

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
            "turn": f"{agent_data.search_turns}_llm",
            "llm_input_text": llm_input_text,
            "llm_response_text": llm_response_text,
            "ids":{
                "prompt_ids_len":len(agent_data.prompt_ids),"response_ids_len":len(agent_data.response_ids),"response_mask_len":len(agent_data.response_mask),"response_logprobs_len":len(agent_data.response_logprobs),
                "prompt_ids": agent_data.prompt_ids, "response_ids": agent_data.response_ids,"response_mask": agent_data.response_mask,"response_logprobs": agent_data.response_logprobs
                }
        })

        # Check termination conditions
        # if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
        #     return AgentState.TERMINATED
        
        content, action = self._parse_llm_response(llm_response_text)
        if action == "answer":
            agent_data.final_answer = content
            agent_data.debug_log_data["turns"][-1]["trigger"] = "answer"
            agent_data.debug_log_data["turns"][-1]["trigger_content"] = content
            return AgentState.TERMINATED
        elif action == "search":
            agent_data.search_query = content
            agent_data.debug_log_data["turns"][-1]["trigger"] = "search"
            agent_data.debug_log_data["turns"][-1]["trigger_content"] = content
            return AgentState.PROCESSING_TOOLS
        else:
            text = f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'
            text_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(text, add_special_tokens=False)
            )
            agent_data.response_ids = text_ids
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_text += text
            agent_data.response_mask += [0] * len(text_ids)
            agent_data.response_logprobs += [0] * len(text_ids)
            agent_data.debug_log_data["turns"][-1]["trigger"] = "invalid_action"
            agent_data.debug_log_data["turns"][-1]["trigger_content"] = text
            agent_data.debug_log_data["turns"][-1]["ids"] = {
                "prompt_ids_len":len(agent_data.prompt_ids),"response_ids_len":len(agent_data.response_ids),"response_mask_len":len(agent_data.response_mask),"response_logprobs_len":len(agent_data.response_logprobs),
                "prompt_ids": agent_data.prompt_ids, "response_ids": agent_data.response_ids,"response_mask": agent_data.response_mask,"response_logprobs": agent_data.response_logprobs
                }
            return AgentState.GENERATING


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
        query = agent_data.search_query

        if retrieval_tool_name not in self.tools:
            available_tools = list(self.tools.keys()) if self.tools else []
            error_text = f"[Retrieval tool not configured] Query: {query}. Available tools: {available_tools}\n"
            error_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(error_text, add_special_tokens=False)
            )
            agent_data.response_ids = error_ids
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_text += error_text
            agent_data.response_mask += [0] * len(error_ids)
            agent_data.response_logprobs += [0] * len(error_ids)
            return AgentState.TERMINATED
        
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
                {"query_list": [query]}
            )
            text = tool_response.text
            text = json.loads(text)["result"]  # Assuming the tool returns a JSON array of results
            text = f"<information>{text}</information>\n\n"
            text_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(text, add_special_tokens=False)
            )
            agent_data.response_ids = text_ids
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_text += text
            agent_data.response_mask += [0] * len(text_ids) 
            agent_data.response_logprobs += [0] * len(text_ids)
            agent_data.search_turns+=1
            agent_data.search_query_list.append(query)
            agent_data.debug_log_data["turns"].append({
                "turn": f"{agent_data.search_turns}_retrieval",
                "search_query": query,
                "retrieval_response_text": text,
                "ids":{
                    "prompt_ids_len":len(agent_data.prompt_ids),"response_ids_len":len(agent_data.response_ids),"response_mask_len":len(agent_data.response_mask),"response_logprobs_len":len(agent_data.response_logprobs),
                    "prompt_ids": agent_data.prompt_ids, "response_ids": agent_data.response_ids,"response_mask": agent_data.response_mask,"response_logprobs": agent_data.response_logprobs
                    }
            })            
            return AgentState.GENERATING
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            agent_data.debug_log_data["turns"].append({
                "turn": f"{agent_data.search_turns}_retrieval",
                "search_query": query,
                "retrieval_response_text": f"Retrieval failed: {e}",
                "ids":{
                    "prompt_ids_len":len(agent_data.prompt_ids),"response_ids_len":len(agent_data.response_ids),"response_mask_len":len(agent_data.response_mask),"response_logprobs_len":len(agent_data.response_logprobs),
                    "prompt_ids": agent_data.prompt_ids, "response_ids": agent_data.response_ids,"response_mask": agent_data.response_mask,"response_logprobs": agent_data.response_logprobs
                    }
            })  
            return AgentState.TERMINATED
        finally:
            if tool and instance_id:
                try:
                    await tool.release(instance_id)
                except Exception:
                    pass
