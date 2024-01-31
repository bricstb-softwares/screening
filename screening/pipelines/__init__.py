
__all__ = ["get_task"]




def get_task( name, process_name ):
    if name=='convnets':
        logger.info(f"getting convnets tasks with process {process}")
        from screening.pipelines.convnets import process
        return process[process_name]
    elif name=='oneclass-svm':
        logger.info(f"getting one class svm tasks with process {process}")
        from screening.pipelines.svm import process
        return process[process_name]