// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/pybind/ir/ir_context.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace pybind {
void IRContextNode::EnterWithContext() {
  VLOG(-2) << this->__ref_count__.to_string();
  IRBuilder::CurrentIRBuilder()->contexts.emplace_back(this);
  VLOG(-2) << this->__ref_count__.to_string();
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
  LinkToParentContext(
      ir::For::Make(loop_var, min, extent, ir::Block::Make(exprs)));
}

void LowerFuncContextNode::ExitWithContext() {
  IRContextNode::ExitWithContext();
  // TODO(6clc): implement Private Fields for intrinstic function, like
  // allreduce
  ir::LoweredFunc lower_func =
      ir::_LoweredFunc_::Make(name, args, ir::Block::Make(exprs));
  IRBuilder ir_builder = IRBuilder::CurrentIRBuilder();
  ir_builder->result = lower_func.operator Expr();
}

Expr IRBuilderNode::Get() const {
  CHECK(result.defined()) << "No result generated in IRBuilder";
  return result;
}

IRBuilder::IRBuilder() {
  common::Shared<IRBuilderNode> n(new IRBuilderNode());
  p_ = n.get();
  p_->contexts.clear();
  p_->result.Reset();
  VLOG(-2) << p_->__ref_count__.to_string();
}

void IRBuilder::EnterWithContext() {
  IRBuilderNode* node = this->get();
  CHECK(node->contexts.empty())
      << "There are still Contexts in IRBuilder that has not been fully converted. \
                                    Please build a new IR with the new IRbuilder";
  node->result.Reset();
  VLOG(-2) << p_->__ref_count__.to_string();
  std::vector<IRBuilder>* st = IRBuilderStack();
  st->push_back(*this);
  VLOG(-2) << p_->__ref_count__.to_string();
}

void IRBuilder::ExitWithContext() {
  VLOG(-2) << p_->__ref_count__.to_string();
  std::vector<IRBuilder>* st = IRBuilderStack();
  CHECK(!st->empty());
  st->pop_back();
  VLOG(-2) << p_->__ref_count__.to_string();
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
    IRContext ir_context = ir_builder->contexts.back();
    ir_context.add_expr(expr);
  }
}

}  // namespace pybind
}  // namespace cinn
