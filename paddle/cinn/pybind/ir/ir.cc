#include "paddle/cinn/pybind/ir/ir.h"
#include "paddle/cinn/pybind/ir/ir_context.h"
namespace cinn {
namespace pybind {
void TensorStore(Expr tensor, Expr value, const std::vector<Expr>& indices) {
  // TODO(6clc): Check the compatibility of data types for tensor and value
  LinkToParentContext(ir::Store::Make(tensor, value, indices));
}
std::vector<Var> AxisMap(std::string kinds, std::vector<Expr> iter_expression) {
  std::vector<Var> rets;
  CHECK_EQ(kinds.size(), iter_expression.size());
  int n = iter_expression.size();
  rets.reserve(n);
  for (int i = 0; i < n; i++) {
    char c = kinds.c_str()[i];

    // TODO(6clc): set bound of IterVar

    Var iter_var = ir::_Var_::Make("", common::Int(32));
    if (c == 'S') {
      iter_var->is_reduce_axis = false;
    } else if (c == 'R') {
      iter_var->is_reduce_axis = true;
    } else {
      LOG(FATAL)
          << "kind of axis setting error, must be R(Reduce) or S(Spatial)";
    }
    rets.push_back(SetScheduleBlockIterVar(iter_var, iter_expression[i]));
  }
}
Var SetScheduleBlockIterVar(Var iter_var, Expr expr) {
  IRContext cur_context =
      IRBuilder::CurrentIRBuilder()->GetLastContext<ScheduleBlockContextNode>();
  ScheduleBlockContextNode* cur_context_node =
      cur_context.As<ScheduleBlockContextNode>();
  cur_context_node->iter_vars.push_back(iter_var);
  cur_context_node->iter_values.push_back(expr);
  return iter_var;
}

Expr Arg(std::string name, Var var) {
  IRContext ctx =
      IRBuilder::CurrentIRBuilder()->FindContext<LowerFuncContextNode>();
  var->name = name;
  ctx.As<LowerFuncContextNode>()->args.emplace_back(var,
                                                    ir::Argument::IO::kUnknown);
  return var.operator Expr();
}

Expr Arg(std::string name, ir::Buffer buffer) {
  IRContext ctx =
      IRBuilder::CurrentIRBuilder()->FindContext<LowerFuncContextNode>();
  buffer->name = "_" + name;
  // TODO(6clc): Unify cinn compilation and runtime Type,
  //  and add a Handle type to Var
  ctx.As<LowerFuncContextNode>()->args.emplace_back(buffer,
                                                    ir::Argument::IO::kUnknown);
  return buffer.operator Expr();
}

IRContext Sequential(Expr min, Expr extent) {
  ForContextNode* for_ctx_node = new ForContextNode();
  for_ctx_node->min = min;
  for_ctx_node->extent = extent;
  for_ctx_node->loop_var = ir::_Var_::Make("v", common::Int(32));
  return IRContext(for_ctx_node);
}

}  // namespace pybind

}  // namespace cinn
