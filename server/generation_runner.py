from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from .decode_runtime import (
    run_decode_step_logits,
    sample_greedy_from_logits,
    sample_temperature_top_p_from_logits,
)
from .generation_types import (
    DebugConfig,
    DecodeStepResult,
    FinishReason,
    GenerationResult,
    GenerationState,
    GreedySampling,
    KvCacheHandle,
    KV_BACKEND_PAGED,
    PrefillResult,
    SamplingConfig,
    TemperatureTopPSampling,
)
from .prefill_runtime import run_prefill_from_input_ids

if TYPE_CHECKING:
    from .inference_session import InferenceSession


class GenerationRunner:
    def __init__(self, inference_session: InferenceSession):
        self.inference_session = inference_session

    def create_generation(
        self,
        request_id: str | None = None,
        sampling_config: SamplingConfig | None = None,
        debug_config: DebugConfig | None = None,
    ) -> GenerationState:
        if request_id is None:
            request_id = self._new_request_id()

        if sampling_config is None:
            sampling_config = SamplingConfig()

        if debug_config is None:
            debug_config = DebugConfig()

        model_name = self._infer_model_name()

        return GenerationState(
            request_id=request_id,
            model_name=model_name,
            sampling_config=sampling_config,
            debug_config=debug_config,
        )

    def prefill(
        self,
        gen: GenerationState,
        input_ids: list[int],
    ) -> PrefillResult:
        if not input_ids:
            raise ValueError("prefill requires non-empty input_ids")

        start_time = time.time()

        gen.reset_for_new_generation()
        gen.prompt_token_ids.extend(input_ids)

        kv_cache = self._allocate_kv_cache(gen=gen, input_ids=input_ids)
        gen.kv_cache = kv_cache

        prefill_outputs = self._run_prefill_forward(
            gen=gen,
            input_ids=input_ids,
        )

        gen.last_logits = self._extract_prefill_last_logits(prefill_outputs)
        gen.is_prefilled = True
        gen.last_token_id = None

        prefill_time_ms = (time.time() - start_time) * 1000.0

        return PrefillResult(
            prompt_tokens=len(input_ids),
            prefill_time_ms=prefill_time_ms,
            per_layer_stats=self._extract_prefill_per_layer_stats(
                gen=gen,
                prefill_outputs=prefill_outputs,
            ),
        )

    def decode_step(
        self,
        gen: GenerationState,
    ) -> DecodeStepResult:
        if not gen.is_prefilled:
            raise RuntimeError("decode_step called before prefill")
        if gen.is_finished:
            raise RuntimeError("decode_step called after generation finished")
        if gen.last_logits is None:
            raise RuntimeError("decode_step requires gen.last_logits")

        step_start_time = time.time()

        token_id = self._sample_from_logits(
            logits=gen.last_logits,
            sampling_config=gen.sampling_config,
        )

        gen.generated_token_ids.append(token_id)
        gen.last_token_id = token_id

        step_text = self._decode_step_text_delta(
            gen=gen,
            token_id=token_id,
        )

        finish_reason = self._check_finish_reason_after_sample(
            gen=gen,
            token_id=token_id,
        )
        if finish_reason is not None:
            gen.is_finished = True
            gen.finish_reason = finish_reason
            gen.last_logits = None

            return DecodeStepResult(
                token_id=token_id,
                text=step_text,
                finish_reason=finish_reason,
                decode_time_ms=(time.time() - step_start_time) * 1000.0,
            )

        decode_outputs = self._forward_one_token(
            gen=gen,
            token_id=token_id,
        )
        gen.last_logits = self._extract_decode_last_logits(decode_outputs)

        return DecodeStepResult(
            token_id=token_id,
            text=step_text,
            finish_reason=None,
            decode_time_ms=(time.time() - step_start_time) * 1000.0,
        )

    def generate(
        self,
        input_ids: list[int],
        sampling_config: SamplingConfig | None = None,
        debug_config: DebugConfig | None = None,
        request_id: str | None = None,
    ) -> GenerationResult:
        generate_started_at = time.time()

        gen = self.create_generation(
            request_id=request_id,
            sampling_config=sampling_config,
            debug_config=debug_config,
        )

        prefill_result = self.prefill(gen=gen, input_ids=input_ids)

        while gen.can_decode:
            self.decode_step(gen)

        generate_finished_at = time.time()

        return self._build_generation_result(
            gen=gen,
            prefill_result=prefill_result,
            generate_started_at=generate_started_at,
            generate_finished_at=generate_finished_at,
        )

    def _new_request_id(self) -> str:
        return uuid.uuid4().hex

    def _infer_model_name(self) -> str:
        cfg = getattr(self.inference_session, "cfg", None)
        if isinstance(cfg, dict):
            model_cfg = cfg.get("model")
            if isinstance(model_cfg, dict):
                name = model_cfg.get("name")
                if isinstance(name, str) and name:
                    return name
        return "unknown_model"

    def _allocate_kv_cache(
        self,
        gen: GenerationState,
        input_ids: list[int],
    ) -> KvCacheHandle:
        cfg = self.inference_session.cfg
        if not isinstance(cfg, dict):
            raise RuntimeError("inference_session.cfg is not a dict")

        kv_cache_cfg = cfg.get("kv_cache")
        if not isinstance(kv_cache_cfg, dict):
            raise RuntimeError("inference_session.cfg['kv_cache'] is not a dict")

        self.inference_session.ensure_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )
        self.inference_session.reset_full_model_kv_cache(
            kv_cache_cfg=kv_cache_cfg,
        )

        handle = self.inference_session.page_attention_cache_managers
        if handle is None:
            raise RuntimeError("page_attention_cache_managers is not initialized")

        return KvCacheHandle(
            backend=KV_BACKEND_PAGED,
            handle=handle,
            capacity_tokens=int(kv_cache_cfg["max_batch_size"]) * int(kv_cache_cfg["max_seq_len"]),
            page_size=int(kv_cache_cfg["page_size"]),
            num_layers=61,
            dtype=str(self.inference_session.backbone_store.dtype)
            if self.inference_session.backbone_store is not None
            else None,
        )

    def _run_prefill_forward(
        self,
        gen: GenerationState,
        input_ids: list[int],
    ) -> Any:
        if gen.kv_cache is None:
            raise RuntimeError("gen.kv_cache is not initialized")

        run_cfg = self.inference_session.cfg["run"]
        start_layer = int(run_cfg["start_layer"])
        end_layer = int(run_cfg["end_layer"])

        return run_prefill_from_input_ids(
            self.inference_session,
            input_ids=input_ids,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache=gen.kv_cache.handle,
            collect_per_layer=bool(gen.debug_config.collect_per_layer),
            prompt=None,
        )

    def _extract_prefill_last_logits(self, prefill_outputs: Any) -> Any:
        if not isinstance(prefill_outputs, dict):
            raise TypeError(
                f"prefill_outputs expected dict, got {type(prefill_outputs).__name__}"
            )

        final_hidden_last_token = prefill_outputs.get("final_hidden_last_token")
        if final_hidden_last_token is None:
            raise RuntimeError(
                "prefill_outputs does not contain 'final_hidden_last_token'"
            )

        executor = self.inference_session.full_model_executor
        if executor is None:
            raise RuntimeError("inference_session.full_model_executor is not initialized")

        logits_result = executor.run_final_norm_and_lm_head(
            final_hidden_last_token,
            return_aux=False,
        )
        logits = logits_result.output
        if logits is None:
            raise RuntimeError("run_final_norm_and_lm_head returned no logits")

        return logits

    def _extract_prefill_per_layer_stats(
        self,
        gen: GenerationState,
        prefill_outputs: Any,
    ) -> Any | None:
        if not gen.debug_config.collect_per_layer:
            return None

        if not isinstance(prefill_outputs, dict):
            raise TypeError(
                f"prefill_outputs expected dict, got {type(prefill_outputs).__name__}"
            )

        return prefill_outputs.get("per_layer")

    def _sample_from_logits(
        self,
        logits: Any,
        sampling_config: SamplingConfig,
    ) -> int:
        strategy = sampling_config.strategy
     
        if isinstance(strategy, GreedySampling):
            return sample_greedy_from_logits(logits)
     
        if isinstance(strategy, TemperatureTopPSampling):
            return sample_temperature_top_p_from_logits(
                logits,
                temperature=float(strategy.temperature),
                top_p=float(strategy.top_p),
            )
     
        raise RuntimeError(
            f"unsupported sampling strategy type: {type(strategy).__name__}"
        )

    def _decode_step_text_delta(
        self,
        gen: GenerationState,
        token_id: int,
    ) -> str | None:
        text = self._decode_output_tokens([token_id])
        if text == "":
            return None
        return text

    def _check_finish_reason_after_sample(
        self,
        gen: GenerationState,
        token_id: int,
    ) -> FinishReason | None:
        # TODO:
        # 1. eos_token
        # 2. stop_token
        # 3. stop_string
        # 4. max_new_tokens
        #
        # The exact priority/order should stay consistent with the chosen
        # generation semantics. For now we only implement stop_token and
        # max_new_tokens.
        if token_id in gen.sampling_config.stop_token_ids:
            return "stop_token"

        if gen.completion_tokens_count >= gen.sampling_config.max_new_tokens:
            return "max_new_tokens"

        return None

    def _forward_one_token(
        self,
        gen: GenerationState,
        token_id: int,
    ) -> Any:
        executor = self.inference_session.full_model_executor
        if executor is None:
            raise RuntimeError("inference_session.full_model_executor is not initialized")

        if gen.kv_cache is None:
            raise RuntimeError("gen.kv_cache is not initialized")

        position_id = gen.prompt_tokens_count + gen.completion_tokens_count - 1
        if position_id < 0:
            raise RuntimeError(f"invalid decode position_id={position_id}")

        current_hidden = executor.embed_token_ids(token_id)

        run_cfg = self.inference_session.cfg["run"]
        start_layer = int(run_cfg["start_layer"])
        end_layer = int(run_cfg["end_layer"])

        return run_decode_step_logits(
            self.inference_session,
            current_hidden=current_hidden,
            position_id=int(position_id),
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache=gen.kv_cache.handle,
        )

    def _extract_decode_last_logits(self, decode_outputs: Any) -> Any:
        if not isinstance(decode_outputs, dict):
            raise TypeError(
                f"decode_outputs expected dict, got {type(decode_outputs).__name__}"
            )

        logits = decode_outputs.get("logits")
        if logits is None:
            raise RuntimeError("decode_outputs does not contain 'logits'")

        return logits

    def _build_generation_result(
        self,
        gen: GenerationState,
        prefill_result: PrefillResult,
        generate_started_at: float,
        generate_finished_at: float,
    ) -> GenerationResult:
        output_token_ids = list(gen.generated_token_ids)
        output_text = self._decode_output_tokens(output_token_ids)

        return GenerationResult(
            request_id=gen.request_id,
            model_name=gen.model_name,
            output_token_ids=output_token_ids,
            output_text=output_text,
            finish_reason=gen.finish_reason,
            prompt_tokens=gen.prompt_tokens_count,
            completion_tokens=gen.completion_tokens_count,
            total_tokens=gen.total_tokens_count,
            generate_started_at=generate_started_at,
            generate_finished_at=generate_finished_at,
            prefill_time_ms=prefill_result.prefill_time_ms,
        )

    def _decode_output_tokens(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""

        tokenizer = self.inference_session.get_deepseek_model_loader().load_tokenizer()
        return tokenizer.decode(token_ids)
