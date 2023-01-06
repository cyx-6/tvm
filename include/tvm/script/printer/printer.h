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
#ifndef TVM_SCRIPT_PRINTER_PRINTER_H_
#define TVM_SCRIPT_PRINTER_PRINTER_H_

#include <tvm/node/node.h>
#include <tvm/script/printer/ir_docsifier.h>

#include <unordered_map>
#include <vector>

namespace tvm {
namespace script {
namespace printer {

String Script(ObjectRef obj, int indent = 4,
              Map<String, String> ir_prefix = {{"ir", "I"}, {"tir", "T"}});

struct Default {
  DataType buffer_dtype = DataType::Float(32);
  DataType int_dtype = DataType::Int(32);
  DataType float_dtype = DataType::Void();

  static Default* Instance();
  static DataType& BufferDType() { return Instance()->buffer_dtype; }
  static DataType& IntDType() { return Instance()->int_dtype; }
  static DataType& FloatDType() { return Instance()->float_dtype; }
};

//////////////////////// CommonAncestorInfo  ////////////////////////

class CommonAncestorInfoNode : public FrameNode {
 public:
  std::unordered_map<const Object*, std::vector<const Object*>> common_prefix;

  void VisitAttrs(AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
    // `common_prefix` is not visited
  }

  static constexpr const char* _type_key = "script.printer.CommonAncestorInfoFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(CommonAncestorInfoNode, FrameNode);
};

class CommonAncestorInfo : public Frame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(CommonAncestorInfo, Frame,
                                                    CommonAncestorInfoNode);
};

CommonAncestorInfo VarUseAnalysis(const IRDocsifier& d, const ObjectRef& root,
                                  runtime::TypedPackedFunc<bool(ObjectRef)> is_var);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_PRINTER_H_
