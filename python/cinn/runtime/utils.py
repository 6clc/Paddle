import inspect


def get_func_global_vars(func):

    if inspect.ismethod(func):
        func = func.__func__

    code = func.__code__
    global_vars = {}
    if func.__closure__ is not None:
        for k, v in zip(code.co_freevars, func.__closure__):
            global_vars[k] = v.cell_contents
    return global_vars


def inspect_function_scope(func):
    scope = {
        **func.__globals__,
        **get_func_global_vars(func),
    }
    return scope
