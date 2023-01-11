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
#include <tvm/runtime/device_api.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

String FindFunctionName(const IRDocsifier& d, const tir::PrimFunc& f) {
  if (!d->mod.defined()) {
    return "main";
  }
  for (const auto& kv : d->mod.value()->functions) {
    if (kv.second.same_as(f)) {
      return kv.first->name_hint;
    }
  }
  return "main";
}

bool IsSimpleBuffer(const tir::Buffer& buf) {
  if (!buf->strides.empty()) {
    return false;
  }
  for (const PrimExpr& shp_i : buf->shape) {
    if (!tir::UndefinedVars(shp_i).empty()) {
      return false;
    }
  }
  for (const PrimExpr& stride_i : buf->strides) {
    if (!tir::UndefinedVars(stride_i).empty()) {
      return false;
    }
  }
  if (!tir::UndefinedVars(buf->elem_offset).empty()) {
    return false;
  } else if (buf->elem_offset->IsInstance<IntImmNode>()) {
    IntImm elem_offset = Downcast<IntImm>(buf->elem_offset);
    if (elem_offset->value != 0) {
      return false;
    }
  }
  return buf.scope() == "global" && buf->data_alignment == runtime::kAllocAlignment &&
         buf->offset_factor == 1 && buf->buffer_type == tir::BufferType::kDefault &&
         !buf->axis_separators.size();
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::PrimFunc>("", [](tir::PrimFunc func, ObjectPath p, IRDocsifier d) -> Doc {
      d->SetCommonPrefix(func, [](const ObjectRef& obj) {
        return obj->IsInstance<tir::VarNode>() || obj->IsInstance<tir::BufferNode>();
      });
      With<TIRFrame> frame(d, func);
      (*frame)->AddDispatchToken(d, "tir");
      int n_args = func->params.size();
      // Step 1. Handle `func->params`
      Array<AssignDoc> args;
      args.reserve(n_args);
      std::unordered_set<const tir::BufferNode*> buffer_inlined;
      for (int i = 0; i < n_args; ++i) {
        tir::Var var = func->params[i];
        ObjectPath var_p = p->Attr("params")->ArrayIndex(i);
        if (func->buffer_map.count(var)) {
          tir::Buffer buffer = func->buffer_map[var];
          ObjectPath buffer_p = p->Attr("buffer_map")->MapValue(var);
          if (IsSimpleBuffer(buffer)) {
            args.push_back(AssignDoc(DefineBuffer(buffer, *frame, d), NullOpt,
                                     BufferAttn(buffer, buffer_p, *frame, d)));
            buffer_inlined.insert(buffer.get());
            continue;
          }
        }
        ExprDoc a = d->AsDoc<ExprDoc>(var->type_annotation, var_p->Attr("type_annotation"));
        args.push_back(AssignDoc(DefineVar(var, *frame, d), NullOpt, a));
      }
      // Step 2. Handle `func->attrs`
      if (func->attrs.defined() && !func->attrs->dict.empty()) {
        (*frame)->stmts.push_back(
            ExprStmtDoc(TIR(d)
                            ->Attr("func_attr")  //
                            ->Call({d->AsDoc<ExprDoc>(func->attrs, p->Attr("attrs"))})));
      }
      // Step 3. Handle `func->buffer_map`
      for (int i = 0; i < n_args; ++i) {
        tir::Var param = func->params[i];
        if (func->buffer_map.count(param)) {
          tir::Buffer buffer = func->buffer_map[param];
          if (buffer_inlined.count(buffer.get())) {
            LOG_INFO << param->name_hint;
            continue;
          }
          ExprDoc param = args[i]->lhs;
          ObjectPath buffer_p = p->Attr("buffer_map")->MapValue(param);
          ExprDoc lhs =
              DefineBuffer(buffer, *frame, d);  // TODO(@junrushao): switch `lhs` and `rhs`
          ExprDoc rhs = BufferDecl(buffer, "match_buffer", {param}, buffer_p, *frame, d);
          (*frame)->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
        }
      }
      // Step 4. Handle `func->body`
      AsDocBody(func->body, p->Attr("body"), frame->get(), d);
      return FunctionDoc(
          /*name=*/IdDoc(FindFunctionName(d, func)),
          /*args=*/args,
          /*decorators=*/{TIR(d)->Attr("prim_func")},
          /*return_type=*/d->AsDoc<ExprDoc>(func->ret_type, p->Attr("ret_type")),
          /*body=*/(*frame)->stmts);
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
