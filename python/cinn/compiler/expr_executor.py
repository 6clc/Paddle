
import ast


class ExprExecutor(object):
    def __init__(self, var_table):
        self.var_table = var_table

    def exec(self, node):
        ret = self.visit(node)
        return ret

    def visit(self, node):
        if isinstance(node, list):
            return [self.visit(item) for item in node]
        if isinstance(node, tuple):
            return (self.visit(item) for item in node)
        assert isinstance(node, ast.AST)
        if isinstance(node, ast.Name):
            return node

        if isinstance(node, ast.Constant):
            return node

        if isinstance(node, (ast.Lambda, ast.Starred)):
            raise Exception("Current not suporrted: Lambda, Starred")

        cls_fields = {}
        for field in node.__class__._fields:
            attr = getattr(node, field)
            if isinstance(attr, (ast.AST, tuple, list)):
                cls_fields[field] = self.visit(attr)
            else:
                cls_fields[field] = attr

        node_type_name = f'eval_{type(node).__name__}'
        if hasattr(self, node_type_name):
            exec_func = getattr(self, node_type_name)
            exec_func(cls_fields)
        else:
            new_node = node.__class__(**cls_fields)
            ast.copy_location(new_node, node)
            new_node = ast.Expression(new_node)
            self.exec_expr(new_node)
            # self.exec_expr(node.__class__(**cls_fields))

    def exec_expr(self, node):
        # if isinstance(node, ast.expr):
        #     node = ast.Expression(body=node)
        # assert isinstance(node, ast.Expression)
        # node = ast.fix_missing_locations(node)
        exec = compile(node, filename="<ast>", mode="eval")
        return eval(exec, self.var_table)
