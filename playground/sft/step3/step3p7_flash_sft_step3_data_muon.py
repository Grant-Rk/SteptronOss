"""Step3.7 Flash text SFT with Muon over the 0311 Step3.7-tokenized shards."""

from playground.data.sft.oss260312.step_sft_data_config0311_step3p7_tokenizer import (
    Recipe0311CompiledStep3p7SFTDataConfig,
)
from playground.pretrain.step3p7.step3p7_flash import Step3p7FlashModelConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from playground.sft.step3.muon_optimizer import Step3p5MuonConfig
from steptronoss.core.parallel_state import PM, get_vpp_size
from steptronoss.exp.base_exp import GradientManagerConfig
from steptronoss.exp.optimizer import AdamConfig
from steptronoss.exp.lr_schedulers import CosineSchedulerConfig
from steptronoss.exp.ntp import MoePretrainMetricConfig
from steptronoss.exp.resources import TorchrunResourceConfig

STEP3P7_FLASH_PATH = "/data/plms/stepfun-ai/Step-3.7-Flash"


class Step3p7F128kSFTResourceConfig(TorchrunResourceConfig):
    def __init__(self):
        super().__init__()
        # Four machines, with eight GPUs launched on each machine.
        self.replica = 4
        self.gpu = 8
        self.envs |= {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }


class Step3p7MuonConfig(Step3p5MuonConfig):
    """Muon grouping for the Step3.7 shared Step3 decoder architecture."""

    def __init__(self):
        super().__init__()
        # Disable Nesterov momentum to reduce the Muon optimizer's memory use.
        self.muon_nesterov = False
        # Fewer Newton-Schulz iterations reduce Muon's temporary peak memory.
        self.muon_ns_steps = 4


class Step3p7MuonGradientManagerConfig(GradientManagerConfig):
    optimizer_cfg = Step3p7MuonConfig

    # ZeRO-3 shards model parameters in addition to gradients and optimizer
    # state. Keep it enabled for the 195B-parameter model even though the
    # PP=2 layout is fully model-parallel (DP=1).
    zero_stage = 3
    """Use ZeRO-3 with Muon for Step3.7 two-dimensional parameters."""


class Step3p7LoRAAdamConfig(AdamConfig):
    """AdamW settings for low-rank adapter parameters."""

    def __init__(self):
        super().__init__()
        self.weight_decay_on_1d_params = False


class Step3p7LoRAGradientManagerConfig(GradientManagerConfig):
    """ZeRO-3 manager used when only LoRA adapters are trainable."""

    optimizer_cfg = Step3p7LoRAAdamConfig
    zero_stage = 3


class Step3p7FlashModelConfigBalanced(Step3p7FlashModelConfig):
    """Step3.7 PP=2 layout across the 32 ranks used by this job."""

    lora_rank: int = 16#0[MODIFY]
    """LoRA rank; zero keeps the original full-parameter training mode."""
    lora_alpha: float = 32.0#16.0#[MODIFY]
    """LoRA scaling numerator."""
    lora_dropout: float = 0.05
    """Dropout applied before the LoRA A projection."""
    lora_targets: tuple[str, ...] = ("wqkv", "wo", "w1", "w2")
    """Linear child names receiving adapters."""

    def __init__(self):
        super().__init__()
        # The job runs on 4 nodes x 8 GPUs. PP=2, TP=8/CP=2 and EP=16 make
        # both attention MP (PP * TP * CP) and MoE MP (PP * ETP * EP) equal
        # WORLD_SIZE=32. TP stays at 8 because Step3.7 has 8 KV heads.
        self.parallel_cfg.pipeline_model_parallel_size = 2
        # Use the non-interleaved PP scheduler for the first multi-node run.
        # The interleaved VPP schedule was stalling during its initial
        # warmup/P2P exchange even though all ranks had reached the barrier.
        self.parallel_cfg.virtual_pipeline_model_parallel_size = 1
        self.parallel_cfg.context_parallel_size = 2
        self.parallel_cfg.tensor_model_parallel_size = 8
        self.parallel_cfg.expert_model_parallel_size = 16
        self.parallel_cfg.expert_tensor_parallel_size = 1
        self.tp_cfg.sequence_parallel = True

    def post_load_model(self, model):
        if self.lora_rank:
            from steptronoss.model.lora import inject_lora

            inject_lora(
                model,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                targets=self.lora_targets,
            )

    def pp_vp_allocation(self, abs_pp_rank: int) -> list[dict]:
        slot_count = PM.size_of("PP") * get_vpp_size()
        if slot_count <= 0:
            raise ValueError(f"invalid PP/VPP slot count: {slot_count}")

        # Interleaved pipeline scheduling requires every virtual chunk to
        # contain a real transformer block.  An empty final chunk still takes
        # part in the send/recv schedule and leaves the ranks out of sync.
        quotient, remainder = divmod(self.num_layers, slot_count)
        if quotient < 1:
            raise ValueError(
                f"num_layers ({self.num_layers}) must be at least the PP/VPP slot count ({slot_count})"
            )
        lengths = [quotient + (index < remainder) for index in range(slot_count)]

        return [{}] * lengths[abs_pp_rank]


class Exp(BaseExp):
    log_dir = "//projects/SteptronOss/logs"

    scheduler_cfg = CosineSchedulerConfig
    resource_cfg = Step3p7F128kSFTResourceConfig
    model_cfg = Step3p7FlashModelConfigBalanced
    metric_cfg = MoePretrainMetricConfig
    data_cfg = Recipe0311CompiledStep3p7SFTDataConfig
    optimizer_cfg = Step3p7MuonGradientManagerConfig

    def __init__(self):
        super().__init__()
        # Set this to a positive rank (for example, 16) for LoRA training.
        # The default remains the original full-parameter Muon run.
        self.model_cfg.lora_rank = 16#0
        if self.model_cfg.lora_rank:
            self.optimizer_cfg = Step3p7LoRAGradientManagerConfig()
            self.suffix = "lora"

    def train(self):
        # Configure defaults before sanity_check(). Model properties such as
        # pp_comm_shape dereference trainer_cfg values during validation.
        self.trainer_cfg.micro_batch_size = 1
        # DP remains one in this fully model-parallel layout. Sixteen
        # microbatches provide enough work to keep both pipeline stages busy.
        self.trainer_cfg.global_batch_size = 16
        # The compiled shards contain samples longer than 1024 tokens and the
        # data config drops oversize samples.  Keep the model's intended pack
        # length so the dataloader does not become empty.
        self.trainer_cfg.global_seq_length = 5120#8192  # 1024 * 128
        self.trainer_cfg.train_iters = 4#5#None
        self.scheduler_cfg.lr = 1e-5
        self.scheduler_cfg.min_lr = 5e-6
        # Keep warmup disabled for the short multimodal smoke run.
        self.scheduler_cfg.warmup_schedule = 0
        self.scheduler_cfg.scheduler_unit = "iter"
        self.scheduler_cfg.weight_decay = 0.1
        self.scheduler_cfg.total_schedule = None
        self.trainer_cfg.log_interval = 1
        # Counting nonzero elements materializes a large temporary tensor for
        # the per-rank gradient shard and can OOM after backward. This metric is
        # diagnostic-only, so disable it for the memory-constrained run.
        self.trainer_cfg.log_num_zeros_in_grad = False

        self.checkpoint_cfg.load_safetensors = STEP3P7_FLASH_PATH
        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_lora_only = bool(self.model_cfg.lora_rank)
        self.checkpoint_cfg.save_dir = "/data/projects/SteptronOss/checkpoints/"
        if self.checkpoint_cfg.save_lora_only:
            self.checkpoint_cfg.save_option.none()
            self.checkpoint_cfg.auto_resume = False
        else:
            self.checkpoint_cfg.save_option.all()
            self.checkpoint_cfg.auto_resume = True
        self.checkpoint_cfg.save_interval = 1

        self.model_cfg.recompute = True
        self.model_cfg.parallel_cfg.context_parallel_size = 2
        self.model_cfg.tp_cfg.sequence_parallel = True
        # Use the pre-profiled batched PP communication path.  The default
        # overlap path creates lazy 2-rank NCCL communicators on the first
        # microbatch and was not making progress in this 32-rank job.
        self.model_cfg.overlap_p2p_comm = True#False
        # HF reshaping runs under the decoder mesh, so the vision encoder must
        # use the same TP width and CP layout as the PP=2/TP=8/CP=2 decoder.
        encoder_parallel_cfg = self.model_cfg.tok_embed_cfg.encoder_cfg.parallel_cfg
        encoder_parallel_cfg.tensor_model_parallel_size = self.model_cfg.parallel_cfg.tensor_model_parallel_size
        encoder_parallel_cfg.context_parallel_size = self.model_cfg.parallel_cfg.context_parallel_size
        # The 80 GiB cards do not have room for bf16 weights plus FP32 grads
        # and optimizer copies of the 1.8B vision encoder. Keep multimodal
        # forward enabled while training the language model.
        self.model_cfg.tok_embed_cfg.encoder_no_grad = True
        self.model_cfg.tok_embed_cfg.projector_no_grad = True
        # Keep saved activations and optimizer state offloaded to preserve
        # headroom for the long sequence and Muon's temporary Newton-Schulz
        # buffers.
        self.model_cfg.pipeline_activation_cpu_offload = True
        self.trainer_cfg.offload_optimizer_state = True
        # Release cached activation blocks before optimizer state is moved
        # back to CUDA. Muon's first step needs an additional 360--720 MiB of
        # temporary workspace, while the allocator may otherwise retain more
        # than 1 GiB of unused cached blocks after backward.
        self.trainer_cfg.empty_unused_memory_level = 3#1
        self.checkpoint_cfg.async_dump = False

        # qwen3_sft_base applies CLI overrides here; select the optimizer
        # afterwards so ``model_cfg.lora_rank=...`` also works from torchrun.
        self.update_from_args()
        self.checkpoint_cfg.save_lora_only = bool(self.model_cfg.lora_rank)
        if self.model_cfg.lora_rank:
            self.checkpoint_cfg.save_option.none()
            self.checkpoint_cfg.auto_resume = False
            self.optimizer_cfg = Step3p7LoRAGradientManagerConfig()
            self.suffix = "lora"
        else:
            self.checkpoint_cfg.save_option.all()
            self.checkpoint_cfg.auto_resume = True
            self.suffix = ""
        self.sanity_check()
        trainer_cls = self.trainer_cfg.get_trainer_cls()
        trainer = trainer_cls(exp=self)
        self.configure_optimizable()
        trainer.train()

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(
            routed_grouped_ffn="fused",
            moe_weighted_gather="triton",
            # A100 does not support the DeepEP/NVSHMEM kernels.  Use the
            # tensorized NCCL all-to-all dispatcher instead; it works across
            # nodes and falls back to regular torch indexing on CUDA.
            TokenDispatcher="alltoall",
            grouped_gemm="nv_grouped_gemm",
            # Use the standard FlashAttention package (FA2), which supports
            # non-Hopper GPUs; FlashAttention-3 requires Hopper hardware.
            AttentionCore="flash-attn",
        )
        # set_optimization(
        #     default="torch_compile",
        #     AttentionCore="flash-attn",
        #     grouped_gemm="nv_grouped_gemm",
        # )


if __name__ == "__main__":
    Exp().train()
