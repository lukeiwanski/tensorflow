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

#include "tensorflow/core/common_runtime/sycl/sycl_device.h"

#include <list>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/platform/tracing.h"

#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
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


Status SYCLDevice::MaybeCopyTensorToGPU(
    const AllocatorAttributes& alloc_attrs, const Tensor& from, Tensor* to,
    StatusCallback done) {
  if (alloc_attrs.on_host()) {
    *to = from;
    done(Status::OK());
    return Status::OK();
  } else {
    if (!DMAHelper::CanUseDMA(&from)) {
      Status err = errors::Internal("SYCL copy from non-DMA ",
                                    DataTypeString(from.dtype()), " tensor");
      done(err);
      return err;
    }
    auto* copy =
        new Tensor(GetAllocator(alloc_attrs), from.dtype(), from.shape());

    // If the tensor is not initialized, we likely ran out of memory.
    if (!copy->IsInitialized()) {
      delete copy;
      Status err = errors::ResourceExhausted(
          "OOM when allocating tensor of shape ", from.shape().DebugString(),
          " and type ", DataTypeString(from.dtype()));
      done(err);
      return err;
    }

    StatusCallback wrapped_done = std::bind(
        [to, copy](StatusCallback done_,
                   // Begin unbound arguments.
                   const Status& s) {
          *to = std::move(*copy);
          delete copy;
          done_(s);
        },
        std::move(done), std::placeholders::_1);

    port::Tracing::ScopedAnnotation annotation("MakeTensorFromProto");
    device_context_->CopyCPUTensorToDevice(&from, this, copy,
                                               std::move(wrapped_done));
    return Status::OK();
  }
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
  if (parsed.dtype() == DT_VARIANT) {
    const Variant* from = parsed.flat<Variant>().data();
    Tensor copy(cpu_allocator(), DT_VARIANT, parsed.shape());
    Variant* copy_variant = copy.flat<Variant>().data();

    std::list<Notification> notifications;
    Status copy_status;
    auto copier = [this, &alloc_attrs, &notifications, &copy_status](
                      const Tensor& from, Tensor* to) {
      // Copier isn't run in a multithreaded environment, so we don't
      // have to worry about the notifications list being modified in parallel.
      notifications.emplace_back();
      Notification& n = *notifications.rbegin();
      return MaybeCopyTensorToGPU(alloc_attrs, from, to,
                                  [&n, &copy_status](const Status& s) {
                                    if (copy_status.ok()) {
                                      copy_status.Update(s);
                                    }
                                    n.Notify();
                                  });
    };
    Status s;
    for (int64 ix = 0; ix < parsed.NumElements(); ++ix) {
      s = VariantDeviceCopy(VariantDeviceCopyDirection::HOST_TO_DEVICE,
                            from[ix], &copy_variant[ix], copier);
      if (!s.ok()) {
        break;
      }
    }
    for (auto& n : notifications) {
      n.WaitForNotification();
    }
    if (!s.ok()) {
      return s;
    }
    *tensor = std::move(copy);
    return copy_status;
  } else {
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

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_SYCL
