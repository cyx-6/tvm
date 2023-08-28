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
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/executable.h>
#include <tvm/runtime/relax_vm/vm.h>

#include <sstream>

#include "./utils.h"
#include "./worker.h"

namespace tvm {
namespace runtime {

class DSOLibraryCache {
 public:
  Module Open(const std::string& library_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    Module& lib = cache_[library_path];
    if (!lib.defined()) {
      lib = Module::LoadFromFile(library_path, "");
    }
    return lib;
  }

  std::unordered_map<std::string, Module> cache_;
  std::mutex mutex_;
};

Module LoadVMModule(std::string path, Device device) {
  static DSOLibraryCache cache;
  Module dso_mod = cache.Open(path);
  device = UseDefaultDeviceIfNone(device);
  PackedFunc vm_load_executable = dso_mod.GetFunction("vm_load_executable");
  CHECK(vm_load_executable != nullptr)
      << "ValueError: File `" << path
      << "` is not built by RelaxVM, because `vm_load_executable` does not exist";
  Module mod = vm_load_executable();
  PackedFunc vm_initialization = mod.GetFunction("vm_initialization");
  CHECK(vm_initialization != nullptr)
      << "ValueError: File `" << path
      << "` is not built by RelaxVM, because `vm_initialization` does not exist";
  vm_initialization(static_cast<int>(device.device_type),                //
                    static_cast<int>(device.device_id),                  //
                    static_cast<int>(relax_vm::AllocatorType::kPooled),  //
                    static_cast<int>(kDLCPU),                            //
                    0,                                                   //
                    static_cast<int>(relax_vm::AllocatorType::kPooled));
  return mod;
}

TVM_REGISTER_GLOBAL("runtime.disco.load_vm_module").set_body_typed(LoadVMModule);

TVM_REGISTER_GLOBAL("runtime.disco.empty").set_body([](TVMArgs args, TVMRetValue* rv) -> void {
  runtime::DataType dtype = args[args.num_args - 2];
  Device device = args[args.num_args - 1];
  int ndim = args.num_args - 2;
  std::vector<ShapeTuple::index_type> shape;
  for (int i = 0; i < ndim; ++i) {
    shape.push_back(args[i].operator int64_t());
  }
  device = UseDefaultDeviceIfNone(device);
  *rv = NDArray::Empty(ShapeTuple(shape), dtype, device);
});

TVM_REGISTER_GLOBAL("runtime.disco.allreduce").set_body([](TVMArgs args, TVMRetValue* rv) -> void {
  std::string ccl = DiscoWorker::ThreadLocal()->ccl;
  std::string pf_name = "runtime.disco." + ccl + ".allreduce";
  const PackedFunc* pf = tvm::runtime::Registry::Get(pf_name);
  CHECK(pf != nullptr) << "ValueError: Cannot find the allreduce function for " << ccl << " via `"
                       << pf_name << "`";
  pf->CallPacked(args, rv);
});

TVM_REGISTER_GLOBAL("runtime.disco.broadcast_from_worker0")
    .set_body([](TVMArgs args, TVMRetValue* rv) -> void {
      std::string ccl = DiscoWorker::ThreadLocal()->ccl;
      std::string pf_name = "runtime.disco." + ccl + ".broadcast_from_worker0";
      const PackedFunc* pf = tvm::runtime::Registry::Get(pf_name);
      CHECK(pf != nullptr) << "ValueError: Cannot find the broadcast function for " << ccl
                           << " via `" << pf_name << "`";
      pf->CallPacked(args, rv);
    });

}  // namespace runtime
}  // namespace tvm
