# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
Planner Agent Loop for End-to-End RL Training (Search-R1 Style)

This follows Search-R1's exact implementation pattern:
1. Maintains two parallel sequences: responses and responses_with_info_mask
2. Uses _info_masked_concatenate_with_padding for mask creation
3. Uses active_mask to track which samples are still generating
4. Supports replan when retrieval fails

Trajectory Structure:
┌─────────────────────────────────────────────────────────────────────────────┐
│ [Prompt] - User question                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ [Response] - LLM generated (mask=1)                                          │
│ "<think>I need to search for...</think><search>query</search>"              │
├─────────────────────────────────────────────────────────────────────────────┤
│ [Information] - Retrieved (mask=0, filled with pad_token_id)                 │
│ "<information>...</information>"                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ [Response] - LLM generated (mask=1)                                          │
│ "<think>Based on info...</think><answer>final answer</answer>"              │
│  OR if retrieval failed:                                                     │
│ "<think>Need to replan...</think><search>new query</search>"                │
└─────────────────────────────────────────────────────────────────────────────┘

Key Output Fields:
- input_ids: [prompt_ids, response_ids] concatenated
- responses: all response tokens (LLM + information)
- responses_with_info_mask: same as responses but information parts filled with pad_token_id
- info_mask: attention_mask style, where information parts are 0
"""
import asyncio
import re
import os
import logging
import torch
from typing import Any, Optional, List, Dict, Tuple
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Special tags (following Search-R1 pattern)
SEARCH_TAG_OPEN = "<search>"
SEARCH_TAG_CLOSE = "</search>"
ANSWER_TAG_OPEN = "<answer>"
ANSWER_TAG_CLOSE = "</answer>"
INFORMATION_TAG_OPEN = "<information>"
INFORMATION_TAG_CLOSE = "</information>"
THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"


@register("planner_r1_agent")
class PlannerR1AgentLoop(AgentLoopBase):
    """
    Planner Agent Loop following Search-R1's exact implementation pattern.
    
    Key differences from original Search-R1:
    1. Uses async interface for tool calls
    2. Supports replan mechanism when retrieval fails
    3. Single sample processing (batch=1) per run() call
    
    The mask mechanism is identical to Search-R1:
    - responses: actual token ids
    - responses_with_info_mask: information tokens replaced with pad_token_id
    - info_mask: 1 for LLM generated, 0 (pad_token_id) for information
    """
    
    MAX_TURNS = 5  # Maximum search turns
    MAX_OBS_LENGTH = 2048  # Maximum observation (information) length
    
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level PlannerR1AgentLoop initialization")
        
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.pad_token_id = tokenizer.pad_token_id
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        
        # Multi-turn config
        multi_turn_config = config.actor_rollout_ref.rollout.multi_turn
        cls.max_turns = getattr(multi_turn_config, 'max_turns', cls.MAX_TURNS)
        cls.max_obs_length = getattr(multi_turn_config, 'max_obs_length', cls.MAX_OBS_LENGTH)
        
        # Initialize search tool
        tool_config_path = multi_turn_config.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        print(f"Initialized tools: {list(cls.tools.keys())}")
        
        # Chat template kwargs
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run the agent loop following Search-R1's pattern.
        
        This implements the same logic as Search-R1's run_llm_loop but for single sample.
        """
        messages = list(kwargs["raw_prompt"])
        request_id = uuid4().hex
        metrics = {}
        tools_kwargs = kwargs.get("tools_kwargs", {})
        
        # ========== Initialize: Tokenize prompt ==========
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
        
        # ========== Initialize state (following Search-R1) ==========
        # left_side: original prompt (for final composition)
        original_left_side = {'input_ids': prompt_ids[:, -self.prompt_length:]}
        
        # right_side: response sequence and its masked version
        # Initially empty tensors with shape [1, 0]
        original_right_side = {
            'responses': prompt_ids[:, 0:0],  # Empty tensor [1, 0]
            'responses_with_info_mask': prompt_ids[:, 0:0]  # Empty tensor [1, 0]
        }
        
        # rolling state: current context for generation
        rollings_input_ids = prompt_ids.clone()
        
        # Tracking variables
        is_done = False
        turn = 0
        total_searches = 0
        found_answer = False
        
        # ========== Main generation loop (following Search-R1) ==========
        while turn < self.max_turns and not is_done:
            turn += 1
            
            # Check if response is too long
            if original_right_side['responses'].shape[1] >= self.response_length:
                break
            
            # Generate LLM response
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=f"{request_id}_turn_{turn}",
                    prompt_ids=rollings_input_ids[0].tolist(),
                    sampling_params=sampling_params,
                )
            
            # Process response (following Search-R1's _postprocess_responses)
            response_ids_list = output.token_ids
            response_text = self.tokenizer.decode(response_ids_list, skip_special_tokens=True)
            
            # Postprocess: stop at </search> or </answer>
            response_text = self._postprocess_response(response_text)
            
            # Re-tokenize processed response
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(response_text, add_special_tokens=False, return_tensors='pt'),
            )
            # response_ids shape: [1, seq_len]
            
            # Parse action (following Search-R1's postprocess_predictions)
            action, content = self._parse_action(response_text)
            
            if action == 'answer':
                # Done! Update right_side with final response (no info)
                original_right_side = self._update_right_side(
                    original_right_side,
                    response_ids,
                    next_obs_ids=None
                )
                is_done = True
                found_answer = True
                
            elif action == 'search':
                # Execute search
                total_searches += 1
                search_result = await self._call_retrieval_tool(content, tools_kwargs)
                
                # Wrap in <information> tags (following Search-R1's execute_predictions)
                next_obs = f"\n\n{INFORMATION_TAG_OPEN}{search_result.strip()}{INFORMATION_TAG_CLOSE}\n\n"
                
                # Tokenize observation
                next_obs_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.encode(next_obs, add_special_tokens=False, return_tensors='pt'),
                )
                
                # Truncate if too long (following Search-R1)
                if next_obs_ids.shape[1] > self.max_obs_length:
                    logger.warning(f"Observation too long: {next_obs_ids.shape[1]} > {self.max_obs_length}")
                    next_obs_ids = next_obs_ids[:, :self.max_obs_length]
                
                # Update right_side with response AND info (info will be masked)
                original_right_side = self._update_right_side(
                    original_right_side,
                    response_ids,
                    next_obs_ids=next_obs_ids
                )
                
                # Update rolling context for next generation
                rollings_input_ids = self._update_rolling_state(
                    rollings_input_ids,
                    response_ids,
                    next_obs_ids
                )
                
            else:
                # Invalid action - add error message and continue
                # Following Search-R1's pattern for invalid actions
                error_msg = (
                    "\nMy previous action is invalid. "
                    "If I want to search, I should put the query between <search> and </search>. "
                    "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
                    "Let me try again.\n"
                )
                next_obs_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.encode(error_msg, add_special_tokens=False, return_tensors='pt'),
                )
                
                # Update right_side (error message is NOT masked, it's feedback)
                original_right_side = self._update_right_side(
                    original_right_side,
                    response_ids,
                    next_obs_ids=next_obs_ids
                )
                
                # Update rolling context
                rollings_input_ids = self._update_rolling_state(
                    rollings_input_ids,
                    response_ids,
                    next_obs_ids
                )
        
        # ========== Final output composition (following Search-R1) ==========
        final_output = self._compose_final_output(
            original_left_side,
            original_right_side
        )
        
        # ========== Build AgentLoopOutput ==========
        # Extract the data we need
        prompt_ids_list = final_output['prompts'][0].tolist()
        response_ids_list = final_output['responses'][0].tolist()
        
        # info_mask is in final_output['info_mask']
        # We need to extract only the response part
        response_length = len(response_ids_list)
        info_mask = final_output['info_mask'][0, -response_length:].tolist()
        
        # Convert info_mask to response_mask format (1 for LLM, 0 for info)
        # In Search-R1, info_mask uses attention_mask style where non-pad is 1
        # But the responses_with_info_mask has pad_token_id where info was
        # So we need to check where info_mask differs from attention_mask
        response_mask = [1 if m != self.pad_token_id else 0 for m in info_mask] # 判断 response_mask 中哪些是真实 token
        
        # Build metrics
        agent_loop_metrics = AgentLoopMetrics(
            generate_sequences=metrics.get("generate_sequences", 0.0),
            tool_calls=metrics.get("tool_calls", 0.0),
        )
        
        output = AgentLoopOutput(
            prompt_ids=prompt_ids_list,
            response_ids=response_ids_list,
            response_mask=response_mask,
            response_logprobs=None,  # Log probs from multi-turn are complex
            num_turns=turn * 2,
            metrics=agent_loop_metrics,
            extra_fields={
                "found_answer": found_answer,
                "total_turns": turn,
                "total_searches": total_searches,
                # Include raw tensors for downstream processing if needed
                "responses_tensor": final_output['responses'],
                "responses_with_info_mask_tensor": final_output['responses_with_info_mask'],
                "info_mask_tensor": final_output['info_mask'],
            },
        )
        
        return output
    
    def _postprocess_response(self, response: str) -> str:
        """
        Process response to stop at </search> or </answer>.
        Following Search-R1's _postprocess_responses logic.
        """
        if SEARCH_TAG_CLOSE in response:
            return response.split(SEARCH_TAG_CLOSE)[0] + SEARCH_TAG_CLOSE
        elif ANSWER_TAG_CLOSE in response:
            return response.split(ANSWER_TAG_CLOSE)[0] + ANSWER_TAG_CLOSE
        return response
    
    def _parse_action(self, response: str) -> Tuple[Optional[str], str]:
        """
        Parse action from response text.
        Following Search-R1's postprocess_predictions logic.
        
        Returns:
            (action, content) where action is 'search', 'answer', or None
        """
        pattern = r'<(search|answer)>(.*?)</\1>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            action = match.group(1)
            content = match.group(2).strip()
            return action, content
        return None, ''
    
    def _info_masked_concatenate_with_padding(
        self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor = None,
        pad_to_left: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate tensors and handle padding.
        Creates a mask (info_mask) to cover the information block if it exists.
        
        This is the EXACT logic from Search-R1.
        
        Args:
            prompt: prompt tensor (or accumulated responses)
            prompt_with_mask: prompt tensor with info already masked
            response: new response tensor
            info: information tensor (will be masked in _with_mask version)
            pad_to_left: whether to pad to left
            
        Returns:
            (padded_tensor, padded_tensor_with_info_masked)
        """
        pad_id = self.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        
        if info is not None:
            tensors.append(info)
            # Create info mask: fill with pad_id to mark as "masked"
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        
        # Sort to handle padding (move pad tokens to left or right)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id # 判断哪些位置是填充
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        
        return padded_tensor, padded_tensor_with_info
    
    def _update_right_side(
        self,
        right_side: Dict[str, torch.Tensor],
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Update right side state with new response and optional observation.
        Following Search-R1's _update_right_side logic.
        
        Args:
            right_side: dict with 'responses' and 'responses_with_info_mask'
            cur_responses: current response tensor [1, seq_len]
            next_obs_ids: observation tensor [1, seq_len] or None
            
        Returns:
            Updated right_side dict
        """
        responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
            right_side['responses'],
            right_side['responses_with_info_mask'],
            cur_responses,
            next_obs_ids,  # This will be masked if not None
            pad_to_left=False
        )
        
        # Truncate to max response length
        max_len = min(self.response_length, responses.shape[1])
        
        return {
            'responses': responses[:, :max_len],
            'responses_with_info_mask': responses_with_info_mask[:, :max_len]
        }
    
    def _update_rolling_state(
        self,
        rollings_input_ids: torch.Tensor,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Update rolling state for next generation.
        Following Search-R1's _update_rolling_state logic.
        
        Args:
            rollings_input_ids: current context [1, seq_len]
            cur_responses: current response [1, seq_len]
            next_obs_ids: observation [1, seq_len]
            
        Returns:
            New rolling input_ids [1, seq_len]
        """
        # Simple concatenation (we don't need the complex padding logic for single sample)
        new_input_ids = torch.cat([rollings_input_ids, cur_responses, next_obs_ids], dim=1)
        
        # Truncate from left to fit max_prompt_length
        max_len = self.prompt_length + self.response_length
        if new_input_ids.shape[1] > max_len:
            new_input_ids = new_input_ids[:, -max_len:]
        
        return new_input_ids
    
    def _compose_final_output(
        self,
        left_side: Dict[str, torch.Tensor],
        right_side: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compose final generation output.
        Following Search-R1's _compose_final_output logic.
        
        Args:
            left_side: dict with 'input_ids' (prompt)
            right_side: dict with 'responses' and 'responses_with_info_mask'
            
        Returns:
            Final output dict with all required fields
        """
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs: [prompt, responses]
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask
        prompt_mask = (left_side['input_ids'] != self.pad_token_id).long()
        response_mask = (right_side['responses'] != self.pad_token_id).long()
        final_output['attention_mask'] = torch.cat([prompt_mask, response_mask], dim=1)
        
        # Create info_mask: same as attention_mask but info parts are masked
        # responses_with_info_mask has pad_token_id where info was
        response_info_mask = (right_side['responses_with_info_mask'] != self.pad_token_id).long()
        final_output['info_mask'] = torch.cat([prompt_mask, response_info_mask], dim=1)
        
        # Create position ids
        final_output['position_ids'] = final_output['attention_mask'].cumsum(dim=1) - 1
        final_output['position_ids'] = final_output['position_ids'].clamp(min=0)
        
        return final_output
    
    async def _call_retrieval_tool(self, query: str, tools_kwargs: dict) -> str:
        """Call the search/retrieval tool."""
        retrieval_tool_name = "search"
        
        if retrieval_tool_name not in self.tools:
            available_tools = list(self.tools.keys()) if self.tools else []
            return f"[Retrieval tool not configured] Query: {query}. Available tools: {available_tools}"
        
        tool = None
        instance_id = None
        try:
            tool = self.tools[retrieval_tool_name]
            kwargs = tools_kwargs.get(retrieval_tool_name, {})
            
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
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
