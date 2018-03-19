/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if TENSORFLOW_USE_SYCL

#include <cstdlib>
#include <algorithm>

#include "tensorflow/core/common_runtime/sycl/sycl_device.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

SYCLDevice::~SYCLDevice() {
  if(gpu_device_info_)
    delete gpu_device_info_;
}

void SYCLDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  assert(context);
  if (port::Tracing::IsActive()) {
    // TODO(pbar) We really need a useful identifier of the graph node.
    const uint64 id = Hash64(op_kernel->name());
    port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                         id);
  }
  op_kernel->Compute(context);
}

Allocator* SYCLDevice::GetAllocator(AllocatorAttributes attr) {
  if (attr.on_host())
    return cpu_allocator_;
  else
    return sycl_allocator_;
}

Status SYCLDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                       const AllocatorAttributes alloc_attrs,
                                       Tensor* tensor) {
  AllocatorAttributes attr;
  attr.set_on_host(true);
  Allocator* host_alloc = GetAllocator(attr);

  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(host_alloc, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }
  Status status;
  if (alloc_attrs.on_host()) {
    *tensor = parsed;
  } else {
    Tensor copy(GetAllocator(alloc_attrs), parsed.dtype(), parsed.shape());

    // If the tensor is not initialized, we likely ran out of memory.
    if (!copy.IsInitialized()) {
      return errors::ResourceExhausted(
          "OOM when allocating tensor of shape ", parsed.shape().DebugString(),
          " and type ", DataTypeString(parsed.dtype()));
    }

    device_context_->CopyCPUTensorToDevice(
        &parsed, this, &copy, [&status](const Status& s) { status = s; });
    *tensor = copy;
  }
  return status;
}

Status SYCLDevice::FillContextMap(const Graph* graph,
                                  DeviceContextMap* device_context_map) {
  // Fill in the context map.  It is OK for this map to contain
  // duplicate DeviceContexts so long as we increment the refcount.
  device_context_map->resize(graph->num_node_ids());
  for (Node* n : graph->nodes()) {
    device_context_->Ref();
    (*device_context_map)[n->id()] = device_context_;
  }

  return Status::OK();
}

Status SYCLDevice::Sync() {
  sycl_allocator_->Synchronize();
  if (sycl_allocator_->Ok()) {
    return Status::OK();
  } else {
    return errors::Internal("Unknown error detected on device ", name());
  }
}

void GSYCLInterface::select_specific_platform_vendor_id(GSYCLInterface::device_list_t& device_list, bool& found_device) {
  auto platform_vendor_cstr = std::getenv("TENSORFLOW_SYCL_USE_PLATFORM_NAME_VENDOR_ID");
  if (!platform_vendor_cstr)
    return;

  std::string platform_vendor_str(platform_vendor_cstr);
  auto sep = platform_vendor_str.rfind(':');
  if (sep == std::string::npos || sep == 0 || sep == platform_vendor_str.size() - 1) {
    LOG(WARNING) << "TENSORFLOW_SYCL_USE_PLATFORM_NAME_VENDOR_ID ignored, "
                    "expected format <platform_name>:<vendor_id>";
    return;
  }

  std::string platform_name(platform_vendor_str, 0, sep);
  auto vendor_id = strtol(std::string(platform_vendor_str, sep + 1).c_str(), NULL, 0);

  // Select the device and remove it from device_list to not select it anymore
  select_first_device(device_list, found_device, [&](const cl::sycl::device& d) {
    if (d.get_info<cl::sycl::info::device::vendor_id>() != vendor_id)
      return false;

    auto device_platform_name = d.get_platform().get_info<cl::sycl::info::platform::name>();
    if (device_platform_name.back() == '\0') // Do not compare the last null character
      device_platform_name.erase(device_platform_name.size() - 1);
    return device_platform_name == platform_name;
  });
  if (!found_device) {
    LOG(WARNING) << "TENSORFLOW_SYCL_USE_PLATFORM_NAME_VENDOR_ID ignored, "
                    "could not find a supported device with platform name \""
                 << platform_name << "\" and vendor id " << vendor_id;
  }
}

void GSYCLInterface::select_specific_device_type(GSYCLInterface::device_list_t& device_list, bool& found_device) {
  auto device_type = std::getenv("TENSORFLOW_SYCL_USE_DEVICE_TYPE");
  if (!device_type)
    return;

  std::string device_type_str(device_type);
  std::transform(device_type_str.begin(), device_type_str.end(), device_type_str.begin(), ::tolower);

  if (device_type_str == "gpu")
    select_first_device(device_list, found_device, [](const cl::sycl::device& d) { return d.is_gpu(); });
  else if (device_type_str == "cpu")
    select_first_device(device_list, found_device, [](const cl::sycl::device& d) { return d.is_cpu(); });
  else if (device_type_str == "acc")
    select_first_device(device_list, found_device, [](const cl::sycl::device& d) { return d.is_accelerator(); });
  else {
    LOG(WARNING) << "TENSORFLOW_SYCL_USE_DEVICE_TYPE ignored, expected value \"gpu\", \"cpu\" or \"acc\"";
    return;
  }

  if (!found_device) {
    LOG(WARNING) << "TENSORFLOW_SYCL_USE_DEVICE_TYPE ignored, "
                    "could not find a supported device of type " << device_type_str;
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_SYCL
