
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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void ScatterKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& index,
                   const phi::DenseTensor& updates,
                   bool overwrite,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("scatter");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    std::vector<int64_t> input_shape = phi::vectorize(x.dims());
    std::vector<int64_t> updates_shape = phi::vectorize(updates.dims());
    std::vector<int64_t> index_shape = phi::vectorize(index.dims());

    int64_t num_indices = index_shape.size();

    if (num_indices == 0) {
      PADDLE_THROW(
          phi::errors::InvalidArgument("num_indices should greater than 0."));
    }

    // If any of the indexed dimensions are zero in the input shape,
    // the update cannotsucceed since it updates a slice of size 1.
    for (int64_t i = 0; i < input_shape.size(); ++i) {
      PADDLE_ENFORCE(
          input_shape.at(i) != 0,
          phi::errors::InvalidArgument("Scatter dimension ", i, " is zero."));
    }
    phi::DenseTensor input_index = MaybeCreateOrTrans64To32bits(dev_ctx, index);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_updates =
        MaybeCreateOrTrans64To32bits(dev_ctx, updates);
    phi::DenseTensor output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);

    LAUNCH_TOPSATENOP(topspaddleScatter,
                      dev_ctx,
                      output,
                      input_x,
                      input_index,
                      input_updates,
                      overwrite);

    MaybeTransResult(dev_ctx, output, out);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Ids"] = {"index"};
    input_names["Updates"] = {"updates"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Ids"] = {const_cast<DenseTensor*>(&index)};
    inputs["Updates"] = {const_cast<DenseTensor*>(&updates)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["overwrite"] = overwrite;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "scatter", dev_ctx);
  }
}
}  // namespace custom_kernel

// PD_REGISTER_PLUGIN_KERNEL(scatter,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::ScatterKernel,
//                           float,
//                           int64_t,
//                           int,
//                           phi::dtype::bfloat16,
//                           phi::dtype::float16) {}
