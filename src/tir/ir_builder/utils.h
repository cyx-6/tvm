/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_TIR_IR_BUILDER_UTILS_H_
#define TVM_TIR_IR_BUILDER_UTILS_H_

#include <tvm/tir/ir_builder.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace ir_builder {
namespace tir {

inline void AddToParent(tvm::tir::Stmt stmt) {
  IRBuilder builder = IRBuilder::Current();
  if (builder->frames.empty()) {
    ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
    builder->result = stmt;
  } else if (const auto* tir_frame = builder->frames.back().as<TIRFrameNode>()) {
    GetRef<TIRFrame>(tir_frame)->stmts.push_back(stmt);
  } else {
    LOG(FATAL) << "TypeError: Unsupported frame type: " << builder->frames.back();
  }
}

inline tvm::tir::Stmt AsStmt(const Array<tvm::tir::Stmt>& stmt) {
  using namespace tvm::tir;
  if (stmt.empty()) {
    return tvm::tir::Evaluate(0);
  } else if (stmt.size() == 1) {
    return stmt[0];
  } else {
    return SeqStmt(stmt);
  }
}

inline BlockFrame FindBlockFrame(const String& method) {
  if (Optional<BlockFrame> frame = IRBuilder::Current()->GetLastFrame<BlockFrame>()) {
    return frame.value();
  }
  LOG(FATAL) << "ValueError: Block frame not find. Please ensure '" << method
             << "' is called under T.block()";
  throw;
}

inline PrimFuncFrame FindPrimFuncFrame(const String& method) {
  if (Optional<PrimFuncFrame> frame = IRBuilder::Current()->GetLastFrame<PrimFuncFrame>()) {
    return frame.value();
  }
  LOG(FATAL) << "ValueError: PrimFunc frame not find. Please ensure '" << method
             << "' is called under T.prim_func()";
  throw;
}

inline IfFrame FindIfFrame(const String& method) {
  if (Optional<IfFrame> frame = IRBuilder::Current()->GetLastFrame<IfFrame>()) {
    return frame.value();
  } else {
    LOG(FATAL) << "ValueError: IfThenElse frame not find. Please ensure '" << method
               << "' is called under T.if_()";
  }
  throw;
}

inline tvm::tir::BufferRegion BufferRegionFromLoad(tvm::tir::BufferLoad buffer_load) {
  Array<Range> ranges;
  for (const PrimExpr& index : buffer_load->indices) {
    ranges.push_back(Range::FromMinExtent(index, IntImm(index->dtype, 1)));
  }
  return tvm::tir::BufferRegion(buffer_load->buffer, ranges);
}

}  // namespace tir
}  // namespace ir_builder
}  // namespace tvm

#endif  // TVM_TIR_IR_BUILDER_UTILS_H_
