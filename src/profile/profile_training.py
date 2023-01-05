import logging

from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from src.models.train_model import main


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        on_trace_ready=tensorboard_trace_handler("./log")
    ) as prof:
        main()
    prof.export_chrome_trace("trace.json")
