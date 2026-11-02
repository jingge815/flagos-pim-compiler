"""PIM 后端的 vLLM V1 执行器：单进程，直接驱动 PIMWorker。"""

from __future__ import annotations

from vllm.v1.executor.uniproc_executor import UniProcExecutor

from backend.vllm_compat import check_vllm_version


class PIMExecutor(UniProcExecutor):
    """把 worker 换成 PIMWorker，其余进程内 RPC 逻辑复用 vLLM 实现。"""

    def _init_executor(self) -> None:
        # 接口是按 0.14 写的，先确认版本再往下走。
        check_vllm_version()
        self.vllm_config.parallel_config.worker_cls = "backend.vllm_worker.PIMWorker"
        super()._init_executor()
