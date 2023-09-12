#include "paddle/cinn/pybind/ir/ir_context.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace pybind {
void IRContextNode::EnterWithContext() {
  IRBuilder::CurrentIRBuilder()->contexts.emplace_back(this);
}
void IRContextNode::ExitWithContext() {
  IRBuilder::CurrentIRBuilder()->contexts.pop_back();
}

void ScheduleBlockContextNode::ExitWithContext() {
  IRContextNode::ExitWithContext();
  ir::Expr schedule_block = ir::ScheduleBlock::Make(
      iter_vars, read_buffers, write_buffers, name, ir::Block::Make(exprs));

  ir::Expr schedule_block_realize =
      ir::ScheduleBlockRealize::Make(iter_values, schedule_block);
  LinkToParentContext(schedule_block_realize);
}

void ForContextNode::ExitWithContext() {
  IRContextNode::ExitWithContext();
  LinkToParentContext(ir::For::Make(loop_var, min, extent, ir::Block::Make(exprs)));
}

Expr IRBuilderNode::Get() const {
  CHECK(result.defined()) << "No result generated in IRBuilder";
  return result;
}

IRBuilder::IRBuilder() {
  p_ = new IRBuilderNode();
  p_->contexts.clear();
  p_->result.Reset();
}

void IRBuilder::EnterWithContext() {
  IRBuilderNode* node = this->get();
  CHECK(node->contexts.empty())
      << "There are still Contexts in IRBuilder that has not been fully converted. \
                                    Please build a new IR with the new IRbuilder";
  node->result.Reset();
  std::vector<IRBuilder>* st = IRBuilderStack();
  st->push_back(*this);
}

void IRBuilder::ExitWithContext() {
  std::vector<IRBuilder>* st = IRBuilderStack();
  CHECK(!st->empty());
  st->pop_back();
}
IRBuilder IRBuilder::CurrentIRBuilder() {
  std::vector<IRBuilder>* st = IRBuilderStack();
  CHECK(!st->empty()) << "No IRBuilder Found";
  return st->back();
}
std::vector<IRBuilder>* IRBuilderStack() {
  thread_local std::vector<IRBuilder> stack;
  return &stack;
}
void LinkToParentContext(ir::Expr expr) {
  IRBuilder ir_builder = IRBuilder::CurrentIRBuilder();
  if (ir_builder->contexts.empty()) {
    ir_builder->result = expr;
  } else {
    const IRContext ir_context = ir_builder->contexts.back();
    ir_context->exprs.push_back(expr);
  }
}

}  // namespace pybind
}  // namespace cinn
