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
#include <tvm/runtime/registry.h>
#include <tvm/script/printer/printer.h>

namespace tvm {
namespace script {
namespace printer {

String Script(ObjectRef obj, int indent, Map<String, String> ir_prefix) {
  IRDocsifier d(ir_prefix);
  Doc doc = d->AsDoc(obj, ObjectPath::Root());
  return DocToPythonScript(doc, indent);
}

Default* Default::Instance() {
  static Default inst;
  return &inst;
}

CommonAncestorInfo VarUseAnalysis(const IRDocsifier& d, const ObjectRef& root,
                                  runtime::TypedPackedFunc<bool(ObjectRef)> is_var) {
  class Visitor : public AttrVisitor {
   public:
    inline void operator()(ObjectRef obj) { Visit("", &obj); }

   private:
    void Visit(const char* key, double* value) final {}
    void Visit(const char* key, int64_t* value) final {}
    void Visit(const char* key, uint64_t* value) final {}
    void Visit(const char* key, int* value) final {}
    void Visit(const char* key, bool* value) final {}
    void Visit(const char* key, std::string* value) final {}
    void Visit(const char* key, void** value) final {}
    void Visit(const char* key, DataType* value) final {}
    void Visit(const char* key, runtime::NDArray* value) final {}
    void Visit(const char* key, ObjectRef* value) final {
      const Object* obj = value->get();
      if (obj == nullptr) {
        return;
      }
      stack_.push_back(obj);
      if (obj->IsInstance<ArrayNode>()) {
        const ArrayNode* array = static_cast<const ArrayNode*>(obj);
        for (ObjectRef element : *array) {
          this->Visit("", &element);
        }
      } else if (obj->IsInstance<MapNode>()) {
        const MapNode* map = static_cast<const MapNode*>(obj);
        for (std::pair<ObjectRef, ObjectRef> kv : *map) {
          this->Visit("", &kv.first);
          this->Visit("", &kv.second);
        }
      } else {
        vtable_->VisitAttrs(const_cast<Object*>(obj), this);
      }
      if (is_var(GetRef<ObjectRef>(obj))) {
        HandleVar(obj);
      }
      stack_.pop_back();
    }

    void HandleVar(const Object* var) {
      if (common_prefix.count(var) == 0) {
        common_prefix[var] = stack_;
        return;
      }
      std::vector<const Object*>& a = common_prefix[var];
      std::vector<const Object*>& b = stack_;
      int n = std::min(a.size(), b.size());
      for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
          a.resize(i);
          break;
        }
      }
    }

    ReflectionVTable* vtable_ = ReflectionVTable::Global();
    std::vector<const Object*> stack_;

   public:
    runtime::TypedPackedFunc<bool(ObjectRef)> is_var;
    std::unordered_map<const Object*, std::vector<const Object*>> common_prefix;
  };
  Visitor visitor;
  visitor.is_var = is_var;
  visitor(root);
  ObjectPtr<CommonAncestorInfoNode> n = make_object<CommonAncestorInfoNode>();
  n->stmts.clear();
  n->d = d.get();
  n->common_prefix = std::move(visitor.common_prefix);
  return CommonAncestorInfo(n);
}

TVM_REGISTER_NODE_TYPE(CommonAncestorInfoNode);
TVM_REGISTER_GLOBAL("script.printer.Script").set_body_typed(Script);
TVM_REGISTER_GLOBAL("script.printer.VarUseAnalysis").set_body_typed(VarUseAnalysis);

}  // namespace printer
}  // namespace script
}  // namespace tvm
