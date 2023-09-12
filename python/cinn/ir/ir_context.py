from .. import core_api


class IRBuilder(object):
    def __init__(self):
        self.ir_builder = core_api.ir.IRBuilder()

    def __enter__(self):
        self.ir_builder.EnterWithContext()
        return self

    def __exit__(self, ptype, value, trace) -> None:  # pylint: disable=unused-argument
        if ptype is None and value is None:
            self.ir_builder.ExitWithContext()

    def get(self):
        return self.ir_builder.get()


class ScheduleBlockContext(object):
    def __init__(self, name):
        self.sch_block_ctx = core_api.ir.ScheduleBlockContext(name)

    def __enter__(self):
        self.sch_block_ctx.EnterWithContext()
        return self

    def __exit__(self, ptype, value, trace) -> None:
        if ptype is None and value is None:
            self.sch_block_ctx.ExitWithContext()
