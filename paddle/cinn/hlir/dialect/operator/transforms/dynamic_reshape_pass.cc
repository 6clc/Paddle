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

#include "paddle/cinn/hlir/dialect/operator/transforms/dynamic_reshape_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class DynamicReshapeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ReshapeOp> {
 public:
  DynamicReshapeOpPattern(
      pir::IrContext* context,
      const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis)
      : pir::OpRewritePattern<paddle::dialect::ReshapeOp>::OpRewritePattern(
            context),
        shape_analysis_(shape_analysis) {}

  bool MatchAndRewrite(paddle::dialect::ReshapeOp op,
                       pir::PatternRewriter& rewriter) const override {
    auto scale_factor_gen_op =
        op->operand_source(1).dyn_cast<pir::OpResult>().owner();
    auto output = op.result(0);

    // The value of shape attribute is fake, we only use the output shape info
    // in shape analysis.
    std::vector<int> shape(
        output.type().dyn_cast<pir::ShapedTypeInterface>().GetRank(), 1);
    shape[0] = -1;

    auto cinn_reshape = rewriter.Build<cinn::dialect::ReshapeOp>(
        op->operand_source(0).dyn_cast<pir::OpResult>(), shape);

    CHECK(shape_analysis_->HasValueShapeDimExprs(output))
        << "Can't find DimExpr for output of reshape in shape_analysis.";
    const auto& out_origin_expr_shape =
        shape_analysis_->GetValueShapeDimExprs(output);
    shape_analysis_->SetValueShapeDimExprs(cinn_reshape.result(0),
                                           out_origin_expr_shape);

    rewriter.ReplaceAllUsesWith(output, cinn_reshape.result(0));
    rewriter.EraseOp(op);

    return true;
  }

 private:
  std::shared_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis_;
};

class DynamicReshapeOpPass : public pir::PatternRewritePass {
 public:
  DynamicReshapeOpPass(
      const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis)
      : PatternRewritePass("cinn_dynamic_reshape_op_pass", /*opt_level=*/1),
        shape_analysis_(shape_analysis) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<DynamicReshapeOpPattern>(context, shape_analysis_);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<cinn::dialect::GroupOp>() && op->num_regions() > 0;
  }

 private:
  std::shared_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis_;
};

std::unique_ptr<pir::Pass> CreateDynamicReshapeOpPass(
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  return std::make_unique<DynamicReshapeOpPass>(shape_analysis);
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
