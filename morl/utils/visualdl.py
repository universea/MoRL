
from visualdl import LogWriter
from morl.utils import logger
from morl.utils.machine_info import get_ip_address

__all__ = []

_writer = None
_WRITTER_METHOD = ['add_scalar', 'add_histogram', 'close', 'flush']


def create_file_after_first_call(func_name):
    def call(*args, **kwargs):
        global _writer
        if _writer is None:
            logdir = logger.get_dir()
            if logdir is None:
                logdir = logger.auto_set_dir(action='d')
                logger.warning(
                    "[VisualDL] logdir is None, will save VisualDL files to {}\nView the data using: visualdl --logdir=./{} --host={}"
                    .format(logdir, logdir, get_ip_address()))
            _writer = LogWriter(logdir=logger.get_dir())
        func = getattr(_writer, func_name)
        func(*args, **kwargs)
        _writer.flush()

    return call


# export writter functions
for func_name in _WRITTER_METHOD:
    locals()[func_name] = create_file_after_first_call(func_name)
    __all__.append(func_name)
