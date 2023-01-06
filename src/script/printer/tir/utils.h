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
#ifndef TVM_SCRIPT_PRINTER_TIR_UTILS_H_
#define TVM_SCRIPT_PRINTER_TIR_UTILS_H_

#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/printer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace script {
namespace printer {

class TIRFrameNode : public FrameNode {
 public:
  ObjectRef tir;
  mutable bool allow_concise_scoping{false};

  void VisitAttrs(AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
    v->Visit("tir", &tir);
    v->Visit("allow_concise_scoping", &allow_concise_scoping);
  }

  static constexpr const char* _type_key = "script.printer.TIRFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(TIRFrameNode, FrameNode);
};

class TIRFrame : public Frame {
 public:
  explicit TIRFrame(const IRDocsifier& d, const ObjectRef& tir) {
    ObjectPtr<TIRFrameNode> n = make_object<TIRFrameNode>();
    n->stmts.clear();
    n->d = d.get();
    n->tir = tir;
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRFrame, Frame, TIRFrameNode);
};

inline IdDoc TIR(const IRDocsifier& p) {  //
  return IdDoc(p->ir_prefix.Get("tir").value_or("T"));
}

inline IdDoc DefineVar(const tir::Var& var, const Frame& frame, const IRDocsifier& d) {
  return d->Define(var, frame, var->name_hint.empty() ? "v" : var->name_hint);
}

inline IdDoc DefineBuffer(const tir::Buffer& buffer, const Frame& frame, const IRDocsifier& d) {
  return d->Define(buffer, frame, buffer->name.empty() ? "buffer" : buffer->name);
}

inline void AddStmtDoc(Array<StmtDoc>* stmts, const Doc& doc) {
  if (const auto* block = doc.as<StmtBlockDocNode>()) {
    for (const StmtDoc& s : block->stmts) {
      stmts->push_back(s);
    }
  } else {
    stmts->push_back(Downcast<StmtDoc>(doc));
  }
}

inline void AsDocBody(const tir::Stmt& stmt, ObjectPath p, TIRFrameNode* f, const IRDocsifier& d) {
  if (const auto* seq_stmt = stmt.as<tir::SeqStmtNode>()) {
    Array<tir::Stmt> body = seq_stmt->seq;
    p = p->Attr("seq");
    for (int i = 0, n = body.size(); i < n; ++i) {
      f->allow_concise_scoping = (i == n - 1);
      AddStmtDoc(&f->stmts, d->AsDoc(body[i], p->ArrayIndex(i)));
    }
  } else {
    f->allow_concise_scoping = true;
    AddStmtDoc(&f->stmts, d->AsDoc(stmt, p));
  }
}

inline Optional<Frame> FindLowestVarDef(const ObjectRef& var, const IRDocsifier& d) {
  int n_frames = d->frames.size();
  std::unordered_map<const Object*, const FrameNode*> tir_to_frame;
  tir_to_frame.reserve(n_frames);
  for (int i = n_frames - 1; i >= 0; --i) {
    if (const auto* f = d->frames[i].as<CommonAncestorInfoNode>()) {
      if (f->common_prefix.count(var.get())) {
        const std::vector<const Object*>& path = f->common_prefix.at(var.get());
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
          if (tir_to_frame.count(*it)) {
            return GetRef<Frame>(tir_to_frame.at(*it));
          }
        }
      }
    } else if (const auto* f = d->frames[i].as<TIRFrameNode>()) {
      tir_to_frame[f->tir.get()] = f;
    }
  }
  return NullOpt;
}

ExprDoc BufferDecl(const tir::Buffer& buffer, const String& method, const Array<ExprDoc>& args,
                   const ObjectPath& p, const Frame& frame, const IRDocsifier& d);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TIR_UTILS_H_
