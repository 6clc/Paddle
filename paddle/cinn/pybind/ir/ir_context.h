#pragma once
#include <map>
#include <vector>
#include "paddle/cinn/common/object.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"

namespace cinn {
namespace pybind {

class IRContextNode : public common::Object {
 public:
  std::vector<Expr> exprs;

 public:
  virtual void EnterWithContext();
  virtual void ExitWithContext();
  const char* type_info() const override { return __type_info__; }

 public:
  static constexpr char* __type_info__ = "IRContextNode";
};

class IRContext : public common::Shared<IRContextNode> {
 public:
  IRContext() = default;
  IRContext(const IRContext& other) : Shared(other.p_) {}
  explicit IRContext(IRContextNode* x) : Shared(x) {}

  template <typename TIRContextNode>
  const TIRContextNode* As() const {
    static_assert(std::is_base_of<IRContextNode, TIRContextNode>());
    CHECK(get()) << "IrContext holds null";
    if (get()->type_info() == TIRContextNode::type_info())
      return static_cast<const TIRContextNode*>(get());
    return nullptr;
  }
  template <typename TIRContextNode>
  TIRContextNode* As() {
    if (get()->type_info() == TIRContextNode::__type_info__)
      return static_cast<TIRContextNode*>(get());
    return nullptr;
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

class IRBuilderNode : public common::Object {
 public:
  std::vector<IRContext> contexts;
  Expr result;
  const char* type_info() const override { return __type_info__; }
  Expr Get() const;

  template <typename TIRContextNode>
  IRContext GetLastContext() const {
    if (TIRContextNode::__type_info__ != contexts.back().get()->type_info()) {
      LOG(FATAL) << "TypeError: The last context is not "
                 << TIRContextNode::__type_info__;
    }
    return contexts.back();
  }

 public:
  static constexpr const char* __type_info__ = "IRBuilderNode";
};
class IRBuilder : public common::Shared<IRBuilderNode> {
 public:
  IRBuilder();
  void EnterWithContext();
  void ExitWithContext();
  static IRBuilder CurrentIRBuilder();
};

std::vector<IRBuilder>* IRBuilderStack();
void LinkToParentContext(ir::Expr);

}  // namespace pybind

}  // namespace cinn
