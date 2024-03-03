
__all__ = ["get_task"]

from loguru import logger


def get_task( name, process_name ):
    if name=='convnets':
        logger.info(f"getting convnets tasks with process {process_name}")
        from screening.pipelines.convnets import processes
        return processes[process_name]
    elif name=='oneclass-svm':
        logger.info(f"getting one class svm tasks with process {process_name}")
        from screening.pipelines.svm import processes
        return processes[process_name]
    else:
        raise RuntimeError(f"train name {name} not supported.")