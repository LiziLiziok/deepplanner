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
        
        # Current context for generation (starts with prompt)
        current_context_ids = list(prompt_ids)
        
        search_turn = 0
        found_answer = False
        
        while search_turn < self.max_search_turns and not found_answer:
            search_turn += 1
            llm_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                       
            # Check if we've exceeded response length
            if len(response_ids) >= self.response_length:
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
            
            # Check for <answer> tag - done!
            if ANSWER_TAG_OPEN in llm_response_text:
                found_answer = True
                break
            
            # Check for <search> tag - need to do retrieval
            search_query = self._extract_search_query(llm_response_text)
            
            if search_query:
                # ========== Call retrieval tool ==========
                retrieval_result = await self._call_retrieval_tool(
                    search_query, 
                    tools_kwargs
                )
                
                # Wrap retrieval result in <information> tags
                information_text = f"{INFORMATION_TAG_OPEN}{retrieval_result}{INFORMATION_TAG_CLOSE}"
                
                # Tokenize retrieval result
                information_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.encode(information_text, add_special_tokens=False),
                )
                
                # Add retrieval result to trajectory with mask=0 (MASKED!)
                response_ids.extend(information_ids)
                response_mask.extend([0] * len(information_ids))  # KEY: mask=0 for retrieval
                response_logprobs.extend([0.0] * len(information_ids))
                
                # Update context for next generation
                current_context_ids.extend(information_ids)
            else:
                # No search tag and no answer tag - might be an error, continue anyway
                logger.warning(f"No <search> or <answer> tag found in response: {llm_response_text[:100]}...")
                # Could add error handling or break here
        
        # Truncate to response_length
        response_ids = response_ids[:self.response_length]
        response_mask = response_mask[:self.response_length]
        response_logprobs = response_logprobs[:self.response_length]
        
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
        
        return output
    
    def _extract_search_query(self, text: str) -> Optional[str]:
        """Extract search query from <search>...</search> tags."""
        pattern = f"{re.escape(SEARCH_TAG_OPEN)}(.*?){re.escape(SEARCH_TAG_CLOSE)}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
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
