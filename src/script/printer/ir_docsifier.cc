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
#include <tvm/runtime/container/base.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/script/printer/ir_docsifier.h>

namespace tvm {
namespace script {
namespace printer {

String GenerateUniqueName(std::string name_hint, std::unordered_set<String>* defined_names) {
  for (char& c : name_hint) {
    if (c != 'c' && !std::isalnum(c)) {
      c = '_';
    }
  }
  std::string name = name_hint;
  for (int i = 1; !defined_names->insert(name).second; ++i) {
    name = name_hint + "_" + std::to_string(i);
  }
  return name;
}

IdDoc IRDocsifierNode::Define(const ObjectRef& obj, const Frame& frame, const String& name_hint) {
  String name = GenerateUniqueName(name_hint, &this->defined_names);
  DocFactory doc_factory = [name]() { return IdDoc(name); };
  auto result = obj2info.insert({obj, VariableInfo{std::move(doc_factory), name}});
  ICHECK(result.second) << "Duplicated object: " << obj;
  IdDoc def_doc(name);
  frame->AddExitCallback([this, obj]() { this->RemoveVar(obj); });
  return def_doc;
}

void IRDocsifierNode::Define(const ObjectRef& obj, const Frame& frame, DocFactory doc_factory) {
  ICHECK(obj2info.find(obj) == obj2info.end()) << "Duplicated object: " << obj;
  ICHECK(!doc_factory()->IsInstance<IdDocNode>())
      << "IRDocsifierNode::Define cannot be used for variable that's mapped to IdDoc.";
  obj2info.insert({obj, VariableInfo{std::move(doc_factory), NullOpt}});
  frame->AddExitCallback([this, obj]() { this->RemoveVar(obj); });
}

Optional<ExprDoc> IRDocsifierNode::GetVarDoc(const ObjectRef& obj) const {
  auto it = obj2info.find(obj);
  if (it == obj2info.end()) {
    return NullOpt;
  }
  return it->second.doc_factory();
}

bool IRDocsifierNode::IsVarDefined(const ObjectRef& obj) const { return obj2info.count(obj); }

void IRDocsifierNode::RemoveVar(const ObjectRef& obj) {
  auto it = obj2info.find(obj);
  ICHECK(it != obj2info.end()) << "No such object: " << obj;
  if (it->second.name.defined()) {
    defined_names.erase(it->second.name.value());
  }
  obj2info.erase(it);
}

IRDocsifier::IRDocsifier(Map<String, String> ir_prefix) {
  auto n = make_object<IRDocsifierNode>();
  n->ir_prefix = std::move(ir_prefix);
  n->dispatch_tokens.push_back("");
  data_ = std::move(n);
}

IRDocsifier::FType& IRDocsifier::vtable() {
  static IRDocsifier::FType inst;
  return inst;
}

TVM_REGISTER_NODE_TYPE(FrameNode);
TVM_REGISTER_GLOBAL("script.printer.FrameAddExitCallback")
    .set_body_typed([](Frame frame, runtime::TypedPackedFunc<void()> callback) {
      frame->AddExitCallback(callback);
    });
TVM_REGISTER_GLOBAL("script.printer.FrameEnterWithScope")
    .set_body_method<Frame>(&FrameNode::EnterWithScope);
TVM_REGISTER_GLOBAL("script.printer.FrameExitWithScope")
    .set_body_method<Frame>(&FrameNode::ExitWithScope);
TVM_REGISTER_NODE_TYPE(IRDocsifierNode);
TVM_REGISTER_GLOBAL("script.printer.IRDocsifier").set_body_typed([](Map<String, String> ir_prefix) {
  return IRDocsifier(ir_prefix);
});
TVM_REGISTER_GLOBAL("script.printer.IRDocsifierPushDispatchToken")
    .set_body_typed([](IRDocsifier p, String token) { p->dispatch_tokens.push_back(token); });
TVM_REGISTER_GLOBAL("script.printer.IRDocsifierPopDispatchToken").set_body_typed([](IRDocsifier p) {
  p->dispatch_tokens.pop_back();
});
TVM_REGISTER_GLOBAL("script.printer.IRDocsifierPushFrame")
    .set_body_typed([](IRDocsifier p, Frame frame) { p->frames.push_back(frame); });
TVM_REGISTER_GLOBAL("script.printer.IRDocsifierPopFrame").set_body_typed([](IRDocsifier p) {
  p->frames.pop_back();
});
TVM_REGISTER_GLOBAL("script.printer.IRDocsifierSetDispatch")
    .set_body_typed([](String token, uint64_t type_index, runtime::PackedFunc f) {
      IRDocsifier::vtable().set_dispatch(token, type_index, std::move(f));
    });
TVM_REGISTER_GLOBAL("script.printer.IRDocsifierRemoveDispatch")
    .set_body_typed([](String token, uint64_t type_index) {
      IRDocsifier::vtable().remove_dispatch(token, type_index);
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
