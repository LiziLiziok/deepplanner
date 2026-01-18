# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
Search-R1 Style Agent Loop for End-to-End RL Training

Trajectory Structure:
┌─────────────────────────────────────────────────────────────────────────────┐
│ [Prompt] - User question                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ [Response Turn1] - LLM generated (mask=1)                                    │
│ "<think>...</think><search>query</search>"                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ [Information] - Retrieved results (mask=0) ⚠️ MASKED                        │
│ "<information>...</information>"                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ [Response Turn2] - LLM generated (mask=1)                                    │
│ "<think>...</think><answer>final answer</answer>"                            │
│  OR if retrieval failed:                                                     │
│ "<think>need replan</think><search>new query</search>"                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ ... continue until <answer> or max turns ...                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Key Features:
1. Single continuous trajectory - all LLM outputs concatenated
2. response_mask: 1 for LLM generated, 0 for retrieved information
3. Special tags: <think>, <search>, <information>, <answer>
4. Retry loop: if retrieval fails, model can generate new search query

"""
import asyncio
import json
import re
import os
import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Special tags for Search-R1 style
SEARCH_TAG_OPEN = "<search>"
SEARCH_TAG_CLOSE = "</search>"
INFORMATION_TAG_OPEN = "<information>"
INFORMATION_TAG_CLOSE = "</information>"
ANSWER_TAG_OPEN = "<answer>"
ANSWER_TAG_CLOSE = "</answer>"
THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"

# Log directory for agent loop
AGENT_LOOP_LOG_DIR = "/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/verl/logs/agent_loop"

# Debug log directory for detailed processing trace
DEBUG_LOG_DIR = "/apdcephfs_szcf/share_303378293/hunyuan/eiraouyang/workplace/paper/verl/logs/search-r1-agent-loop-log"


@register("search_r1_agent")
class SearchR1AgentLoop(AgentLoopBase):
    """
    Search-R1 Style Agent Loop for End-to-End RL Training
    
    This agent loop:
    1. Lets LLM generate with special tags (<think>, <search>, <answer>)
    2. When <search>query</search> is detected, calls retrieval tool
    3. Wraps retrieval results in <information>...</information> and appends to trajectory
    4. Masks retrieval content (mask=0) so it doesn't participate in gradient computation
    5. Continues until <answer> is found or max turns reached
    6. If retrieval fails, LLM can generate new search query (retry mechanism)
    """
    
    MAX_SEARCH_TURNS = 4  # Maximum number of search attempts
    
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level SearchR1AgentLoop initialization")
        
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        
        # Multi-turn config
        cls.max_search_turns = getattr(
            config.actor_rollout_ref.rollout.multi_turn, 
            'max_search_turns', 
            cls.MAX_SEARCH_TURNS
        )
        
        # Initialize search tool
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        print(f"Initialized tools: {list(cls.tools.keys())}")
        
        # Chat template kwargs
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run the Search-R1 style agent loop.
        
        Args:
            sampling_params: LLM sampling parameters
            **kwargs: Dataset fields including raw_prompt, tools_kwargs, etc.
            
        Returns:
            AgentLoopOutput with:
            - prompt_ids: Original prompt token ids
            - response_ids: All generated + retrieved content token ids
            - response_mask: 1 for LLM generated, 0 for retrieved content
        """
        messages = list(kwargs["raw_prompt"])
        question_ = messages[-1]["content"]

        request_id = uuid4().hex
        metrics = {}
        tools_kwargs = kwargs.get("tools_kwargs", {})
        
        # Initialize trajectory tracking
        prompt_ids: list[int] = []  # Original prompt
        response_ids: list[int] = []  # All response tokens (LLM + retrieval)
        response_mask: list[int] = []  # 1 for LLM, 0 for retrieval
        response_logprobs: list[float] = []  # Log probs (0 for retrieval)
        
        # Debug log data collection
        debug_log_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "question": question_,
            "response_length_limit": self.response_length,
            "max_search_turns": self.max_search_turns,
            "turns": [],  # Each turn's detailed info
        }
        
        # Tokenize initial prompt
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        
        debug_log_data["prompt_ids_length"] = len(prompt_ids)
        
        # Current context for generation (starts with prompt)
        current_context_ids = list(prompt_ids)
        
        search_turn = 0
        found_answer = False
        
        while search_turn < self.max_search_turns and not found_answer:
            search_turn += 1
            
            turn_log = {
                "turn": search_turn,
                "response_ids_length_before_turn": len(response_ids),
            }
            
            llm_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                       
            # Check if we've exceeded response length
            if len(response_ids) >= self.response_length:
                turn_log["early_break_reason"] = "response_ids >= response_length"
                debug_log_data["turns"].append(turn_log)
                break
            
            # ========== Generate LLM response ==========
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=f"{request_id}_turn_{search_turn}",
                    prompt_ids=current_context_ids,
                    sampling_params=sampling_params,
                )
            
            llm_response_ids = output.token_ids
            llm_log_probs = output.log_probs or [0.0] * len(llm_response_ids)
            
            # Add LLM response to trajectory with mask=1
            response_ids.extend(llm_response_ids)
            response_mask.extend([1] * len(llm_response_ids))
            response_logprobs.extend(llm_log_probs)
            
            # Update context
            current_context_ids.extend(llm_response_ids)
            
            # Decode to check for tags
            llm_response_text = self.tokenizer.decode(llm_response_ids, skip_special_tokens=True)
            
            # Log LLM generation info
            turn_log["llm_response_ids_length"] = len(llm_response_ids)
            turn_log["llm_response_text"] = llm_response_text
            turn_log["response_ids_length_after_llm"] = len(response_ids)
            
            # Check if LLM hallucinated <information> tags (generated without real retrieval)
            llm_hallucinated_information = INFORMATION_TAG_OPEN in llm_response_text
            if llm_hallucinated_information:
                turn_log["warning_llm_hallucinated_information"] = True
                turn_log["hallucinated_information_in_llm_output"] = llm_response_text
            
            # Extract search query FIRST - even if answer is found, we want to compare with real retrieval
            search_query = self._extract_search_query(llm_response_text)
            turn_log["search_query"] = search_query
            
            # If LLM hallucinated information but also has search query, perform real retrieval for comparison
            if llm_hallucinated_information and search_query:
                try:
                    real_retrieval_result = await self._call_retrieval_tool(
                        search_query, 
                        tools_kwargs
                    )
                    turn_log["real_retrieval_for_comparison"] = real_retrieval_result
                    turn_log["real_retrieval_for_comparison_length"] = len(real_retrieval_result)
                except Exception as e:
                    turn_log["real_retrieval_for_comparison_error"] = str(e)
            
            # Check for <answer> tag - done!
            if ANSWER_TAG_OPEN in llm_response_text:
                found_answer = True
                turn_log["found_answer"] = True
                turn_log["real_retrieval_performed"] = False  # No real retrieval was appended to trajectory
                debug_log_data["turns"].append(turn_log)
                break
            
            if search_query:
                # ========== Call retrieval tool ==========
                retrieval_result = await self._call_retrieval_tool(
                    search_query, 
                    tools_kwargs
                )
                
                # Save question, search_query and retrieval_result to jsonl log file
                await self._save_agent_loop_log(question_, search_query, retrieval_result)
                
                # Wrap retrieval result in <information> tags
                information_text = f"{INFORMATION_TAG_OPEN}{retrieval_result}{INFORMATION_TAG_CLOSE}"
                
                # Log retrieval info - REAL retrieval was performed
                turn_log["real_retrieval_performed"] = True
                turn_log["retrieval_result"] = retrieval_result
                turn_log["retrieval_result_length"] = len(retrieval_result)
                turn_log["information_text"] = information_text
                turn_log["information_text_length"] = len(information_text)
                
                # Tokenize retrieval result
                information_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.encode(information_text, add_special_tokens=False),
                )
                
                turn_log["information_ids_length"] = len(information_ids)
                turn_log["response_ids_length_before_extend_info"] = len(response_ids)
                
                # Add retrieval result to trajectory with mask=0 (MASKED!)
                response_ids.extend(information_ids)
                response_mask.extend([0] * len(information_ids))  # KEY: mask=0 for retrieval
                response_logprobs.extend([0.0] * len(information_ids))
                
                turn_log["response_ids_length_after_extend_info"] = len(response_ids)
                
                # Update context for next generation
                current_context_ids.extend(information_ids)
            else:
                # No search tag and no answer tag - might be an error, continue anyway
                logger.warning(f"No <search> or <answer> tag found in response: {llm_response_text[:100]}...")
                turn_log["warning"] = "No <search> or <answer> tag found"
                # Could add error handling or break here
            
            debug_log_data["turns"].append(turn_log)
        
        # Log before truncation
        debug_log_data["response_ids_length_before_truncate"] = len(response_ids)
        debug_log_data["response_text_before_truncate"] = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Truncate to response_length
        response_ids = response_ids[:self.response_length]
        response_mask = response_mask[:self.response_length]
        response_logprobs = response_logprobs[:self.response_length]
        
        # Log after truncation
        debug_log_data["response_ids_length_after_truncate"] = len(response_ids)
        debug_log_data["response_text_after_truncate"] = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        debug_log_data["was_truncated"] = debug_log_data["response_ids_length_before_truncate"] > self.response_length
        
        # Build metrics
        agent_loop_metrics = AgentLoopMetrics(
            generate_sequences=metrics.get("generate_sequences", 0.0),
            tool_calls=metrics.get("tool_calls", 0.0),
        )
        
        # Build output
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
            num_turns=search_turn * 2,  # Each turn has LLM response + retrieval
            metrics=agent_loop_metrics,
            extra_fields={
                "found_answer": found_answer,
                "search_turns": search_turn,
                "question": question_,
                "response_text": self.tokenizer.decode(response_ids, skip_special_tokens=True),
            },
        )
        
        # Log final output info
        debug_log_data["final_output"] = {
            "found_answer": found_answer,
            "search_turns": search_turn,
            "response_text": output.extra_fields["response_text"],
        }
        
        # Save debug log
        await self._save_debug_log(debug_log_data)
        
        return output
    
    def _extract_search_query(self, text: str) -> Optional[str]:
        """Extract search query from <search>...</search> tags."""
        pattern = f"{re.escape(SEARCH_TAG_OPEN)}(.*?){re.escape(SEARCH_TAG_CLOSE)}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
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
    
    async def _save_agent_loop_log(self, question: str, search_query: str, retrieval_result: str) -> None:
        """
        Save question, search_query and retrieval_result to jsonl log file.
        
        Args:
            question: The original question from user
            search_query: The search query extracted from LLM response
            retrieval_result: The retrieval result from search tool
        """
        try:
            # Ensure log directory exists
            os.makedirs(AGENT_LOOP_LOG_DIR, exist_ok=True)
            
            # Generate log file path with date
            date_str = datetime.now().strftime("%Y%m%d")
            log_file_path = os.path.join(AGENT_LOOP_LOG_DIR, f"agent_loop_log_{date_str}.jsonl")
            
            # Create log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "search_query": search_query,
                "retrieval_result": retrieval_result
            }
            
            # Write to jsonl file (append mode)
            def write_log():
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
            await self.loop.run_in_executor(None, write_log)
            
        except Exception as e:
            logger.error(f"Failed to save agent loop log: {e}")
    
    async def _call_retrieval_tool(self, query: str, tools_kwargs: dict) -> str:
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
            return f"[Retrieval tool not configured] Query: {query}. Available tools: {available_tools}"
        
        tool = None
        instance_id = None
        try:
            tool = self.tools[retrieval_tool_name]
            kwargs = tools_kwargs.get(retrieval_tool_name, {})
            
            # Create tool instance
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            
            # Execute search - SearchTool.execute expects query_list parameter
            tool_response, tool_reward, res = await tool.execute(
                instance_id,
                {"query_list": [query]}
            )
            
            return tool_response.text or ""
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return f"[Retrieval failed] {str(e)}"
        finally:
            if tool and instance_id:
                try:
                    await tool.release(instance_id)
                except Exception:
                    pass
