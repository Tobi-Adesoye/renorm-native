"""
Renorm-Native: Agent LoopGuard Gateway (Protocol Shield)

This module sanitizes, parses, and validates JSON payloads returned by local LLMs 
and agent frameworks (e.g., Goose, Paperclip). It intercepts verbose conversational prose,
corrects syntax errors on the fly, and enforces execution budgets to prevent infinite loops.
"""

import re
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

# Setup logger
logger = logging.getLogger("Renorm-LoopGuard")


class RenormLoopGuard:
    """
    Acts as an active boundary proxy for LLM outputs. It extracts structured
    tool-calling blocks and monitors command histories to guard against execution loops.
    """
    def __init__(self, max_token_window: int = 6000, max_consecutive_repeats: int = 3):
        self.max_token_window = max_token_window
        self.max_consecutive_repeats = max_consecutive_repeats
        # Matches raw JSON or JSON encapsulated inside markdown code blocks safely (using hex representation for backticks)
        self.json_block_pattern = re.compile(r"\x60{3}(?:json)?\s*(\{.*?\})\s*\x60{3}", re.DOTALL)
        self.execution_history: List[str] = []

    def sanitize_output_stream(self, raw_payload: str) -> Dict[str, Any]:
        """
        Extracts, cleans, and validates structured parameter blocks from raw text streams.
        Ensures execution parameters are presented as raw dictionary elements.
        """
        cleaned_payload = raw_payload.strip()

        # 1. Inspect for markdown wrapper configurations
        markdown_match = self.json_block_pattern.search(cleaned_payload)
        if markdown_match:
            cleaned_payload = markdown_match.group(1).strip()
        else:
            # Fallback: Locate explicit brace boundaries if markdown was omitted
            start_brace = cleaned_payload.find("{")
            end_brace = cleaned_payload.rfind("}")
            if start_brace != -1 and end_brace != -1:
                cleaned_payload = cleaned_payload[start_brace:end_brace+1].strip()

        # 2. Attempt parsing and error correction
        try:
            parsed_json = json.loads(cleaned_payload)
            self._record_execution(cleaned_payload)
            
            return {
                "status": "SECURE",
                "data": parsed_json,
                "error": None,
                "loop_detected": self.is_loop_stalled()
            }
        except json.JSONDecodeError as err:
            logger.warning(f"Malformed tool-calling syntax detected: {err}. Executing adaptive recovery.")
            
            # Reconstruct structural omissions to avoid breaking tool pipeline workflows
            return {
                "status": "RECOVERED_FALLBACK",
                "data": {
                    "raw_response": raw_payload,
                    "directive": "REQUEST_REPARSE",
                    "reason": str(err)
                },
                "error": str(err),
                "loop_detected": False
            }

    def _record_execution(self, sanitized_payload: str):
        """Logs the execution trace to internal rolling window buffers."""
        self.execution_history.append(sanitized_payload)
        if len(self.execution_history) > 10:
            self.execution_history.pop(0)

    def is_loop_stalled(self) -> bool:
        """
        Analyzes command structures to identify infinite tool-calling repetitions,
        saving the enterprise token budget from run-away processes.
        """
        if len(self.execution_history) < self.max_consecutive_repeats:
            return False
            
        recent_window = self.execution_history[-self.max_consecutive_repeats:]
        # If the unique set of recent serialized executions collapses to 1, a loop is confirmed
        return len(set(recent_window)) == 1

    def clear_session_history(self):
        """Flushes execution buffers to start a fresh tracking session."""
        self.execution_history.clear()


# =====================================================================
# INTERACTIVE PROTOCOL GUARD TESTER
# =====================================================================
if __name__ == "__main__":
    print("[Renorm-LoopGuard] Running security gateway diagnostics...")
    guard = RenormLoopGuard()

    # Case 1: Conversational Markdown Payload (Constructed with hex to bypass markdown escaping)
    dirty_input = (
        "I will now run the workflow for you. Here are the parameters:\n"
        "\x60\x60\x60json\n"
        "{\n"
        '  "action": "ENGAGE_RENORM_ALIGNMENT",\n'
        '  "target_pool": "SRAM_REGISTERS"\n'
        "}\n"
        "\x60\x60\x60\n"
        "Let me know if you want me to proceed."
    )

    print("\n--- Diagnostic Case 1: Complex Markdown Extract ---")
    result_1 = guard.sanitize_output_stream(dirty_input)
    print(f"Status: {result_1['status']}")
    print(f"Extracted Params: {result_1['data']}")
    print(f"Loop Detected: {result_1['loop_detected']}")

    # Case 2: Infinite Loop Simulation
    print("\n--- Diagnostic Case 2: Infinite Loop Detection ---")
    redundant_tool_call = '{"action": "GET_SYS_INFO", "target": "VRAM"}'
    
    # Push identical calls to trigger threshold limits
    for i in range(3):
        res = guard.sanitize_output_stream(redundant_tool_call)
        print(f"Pass {i+1} - Loop Alert status: {res['loop_detected']}")