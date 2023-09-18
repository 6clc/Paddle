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

#pragma once
#include <map>
#include <vector>
#include "paddle/cinn/common/object.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/lowered_func.h"

namespace cinn {
namespace pybind {

class IRContextNode : public common::Object {
 public:
  std::vector<ir::Expr> exprs;

 public:
  virtual void EnterWithContext();
  virtual void ExitWithContext();
  const char* type_info() const override { return __type_info__; }

 public:
  static constexpr char* __type_info__ = "IRContextNode";
};

class IRContext {
 public:
  IRContext() = default;
  IRContext(const IRContext& other) = default;
  explicit IRContext(IRContextNode* x) : data_(x) {}

  const IRContextNode* get() const { return data_.get(); }
  const IRContextNode* operator->() const { return data_.get(); }

  void add_expr(Expr expr) { data_->exprs.push_back(expr); }

 public:
  common::Shared<IRContextNode> data_;

 public:
  template <typename TIRContextNode>
  const TIRContextNode* As() const {
    static_assert(std::is_base_of<IRContextNode, TIRContextNode>());
    CHECK(data_.get()) << "IrContext holds null";
    auto* ctx_node = data_.get()->safe_as<TIRContextNode>();
    if (!ctx_node) {
      LOG(FATAL) << "TypeConvertError: convert " << data_.get()->type_info()
                 << " to " << TIRContextNode::__type_info__;
    }
    return ctx_node;
  }
  template <typename TIRContextNode>
  TIRContextNode* As() {
    CHECK(data_.get()) << "IrContext holds null";
    auto* ctx_node = data_.get()->safe_as<TIRContextNode>();
    if (!ctx_node) {
      LOG(FATAL) << "TypeConvertError: convert " << data_.get()->type_info()
                 << " to " << TIRContextNode::__type_info__;
    }
    return ctx_node;
  }
};

class ScheduleBlockContextNode : public IRContextNode {
 public:
  std::vector<Var> iter_vars;
  // BufferRange(s) which is read in this schedule block, it is used to
  // analyze, not a real computation expression. Must be AST DFS order.
  std::vector<Expr> read_buffers;
  // BufferRange(s) which is written in this schedule block, it is used to
  // analyze, not a real computation expression. Must be AST DFS order.
  std::vector<Expr> write_buffers;
  // Additional attributes about this schedulable block,
  // which take some auxiliary hints for future transformations.
  std::map<std::string, ir::attr_t> attrs;
  // values of the iter_vars
  std::vector<Expr> iter_values;
  std::string name;

 public:
  ScheduleBlockContextNode() = default;
  ScheduleBlockContextNode(std::string name) : name(name) {}
  void ExitWithContext() final;
  const char* type_info() const override { return __type_info__; }

 public:
  static constexpr const char* __type_info__ = "ScheduleBlockContextNode";
};

class ScheduleBlockContext : public IRContext {
 public:
  ScheduleBlockContext(ScheduleBlockContextNode* x) : IRContext(x) {}
};

class ForContextNode : public IRContextNode {
 public:
  //! The loop variable.
  Var loop_var;
  //! The minimum value of the iteration.
  Expr min;
  //! The extent of the iteration.
  Expr extent;

 public:
  void ExitWithContext() final;
  const char* type_info() const override { return __type_info__; }

 public:
  static constexpr const char* __type_info__ = "ForContextNode";
};

class ForContext : public IRContext {
 public:
  explicit ForContext(ForContextNode* x) : IRContext(x) {}
};

class LowerFuncContextNode : public IRContextNode {
 public:
  //! The name of this function.
  std::string name;
  //! The Arguments used in the body of the function.
  std::vector<ir::Argument> args;

 public:
  LowerFuncContextNode() = default;
  LowerFuncContextNode(std::string name) : name(name) {}
  void ExitWithContext() final;
  const char* type_info() const override { return __type_info__; }

 public:
  static constexpr const char* __type_info__ = "LowerFuncContextNode";
};

class LowerFuncContext : public IRContext {
 public:
  explicit LowerFuncContext(LowerFuncContextNode* x) : IRContext(x) {}
};

class IRBuilderNode : public common::Object {
 public:
  std::vector<IRContext> contexts;
  Expr result;
  const char* type_info() const override { return __type_info__; }
  Expr GetResult() const;
  void Reset();

  template <typename TIRContextNode>
  IRContext GetLastContext() const;

  template <typename TIRContextNode>
  IRContext FindContext() const;

 public:
  static constexpr const char* __type_info__ = "IRBuilderNode";
};
class IRBuilder {
 public:
  IRBuilder();
  void EnterWithContext();
  void ExitWithContext();
  static IRBuilder CurrentIRBuilder();

 public:
  common::Shared<IRBuilderNode> data_;
};

std::vector<IRBuilder>* IRBuilderStack();
void LinkToParentContext(ir::Expr);

template <typename TIRContextNode>
IRContext IRBuilderNode::GetLastContext() const {
  if (!(contexts.back().As<TIRContextNode>())) {
    LOG(FATAL) << "TypeError: The last context is not "
               << TIRContextNode::__type_info__;
  }
  return contexts.back();
}

template <typename TIRContextNode>
IRContext IRBuilderNode::FindContext() const {
  for (auto it = contexts.rbegin(); it != contexts.rend(); ++it) {
    if (const TIRContextNode* p = it->As<TIRContextNode>()) {
      return *it;
    }
  }
}

}  // namespace pybind

}  // namespace cinn
