import torch
import torch.distributed as dist
from typing import Any, Optional, Tuple

try:
    # noinspection PyUnresolvedReferences
    from deep_ep_xpu_cpp import EventHandle
except Exception:  # pragma: no cover - exercised when the native extension is unavailable
    EventHandle = Any  # type: ignore[misc,assignment]


class EventOverlap:
    """
    A thin wrapper around an XPU event for stream-overlap convenience.

    Attributes:
        event: the captured backend event handle.
        extra_tensors: tensors kept alive with the event as a lightweight `record_stream` substitute,
            which is useful for graph capture friendly code paths.
    """

    def __init__(self, event: Optional[EventHandle] = None, extra_tensors: Optional[Tuple[torch.Tensor]] = None) -> None:
        """
        Initialize the class.

        Arguments:
            event: the captured backend event handle.
            extra_tensors: tensors kept alive with the event as a lightweight `record_stream` substitute.
        """
        self.event = event

        # Keep tensors alive until the event has been waited on. This mirrors
        # `record_stream` style lifetime management without depending on it.
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        """
        The current stream `torch.xpu.current_stream()` waits for the event to be finished.
        """
        assert self.event is not None
        self.event.current_stream_wait()

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap:
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream waits for the event.
        ```
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:
            self.event.current_stream_wait()


def check_nvlink_connections(group: dist.ProcessGroup):
    """
    Compatibility no-op retained under the original API name.

    The CUDA wrapper validated NVLink topology from Python. The XPU path relies on
    the runtime/launcher to provide a valid peer-access configuration, so the check
    stays as a lightweight placeholder for API compatibility.

    Arguments:
        group: the communication group.
    """
    if group is None or not torch.xpu.is_available():
        return
