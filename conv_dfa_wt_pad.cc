#include <cnpy.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/NEON/functions/NEReduceMean.h"
#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"
#include "arm_compute/runtime/NEON/functions/NETranspose.h"
#include "arm_compute/runtime/Tensor.h"
#include "device.hpp"

#define HEIGHT 224
#define WIDTH 224
#define CHANNEL 3
#define OUTPUT_CHANNELS 64
#define INPUT_CHANNELS 3
#define KERNEL_SIZE 3
#define ONE_BYTE 8
#define OFFSET 2
#define PADDING 0
#define STRIDE 1

using namespace arm_compute;

// get tensor shape as string
std::string get_tensor_shape(const TensorShape &shape) {
  std::string shape_str = "[";
  for (size_t i = 0; i < shape.num_dimensions() + 1; ++i) {
    shape_str += std::to_string(shape[i]) + " ";
    if (i < shape.num_dimensions()) {
      shape_str += ",";
    }
  }
  shape_str += " ]";
  return shape_str;
}
// save tensor to npy file
void save_tensor_to_npy(const Tensor &tensor, const std::string &filename) {
  Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());
  std::vector<float> outputData;
  execute_window_loop(window, [&](const Coordinates &id) {
    outputData.push_back(
        *reinterpret_cast<const float *>(tensor.ptr_to_element(id)));
  });
  // only copy till the last dimension
  std::vector<size_t> shape{tensor.info()->tensor_shape().begin(),
                            tensor.info()->tensor_shape().begin() +
                                tensor.info()->tensor_shape().num_dimensions() +
                                1};
  std::reverse(shape.begin(), shape.end());
  cnpy::npy_save(filename, &outputData[0], shape, "w");
}
int main() {
  const int total_pe = (PE_ROWS * PE_COLUMNS);
  /* Initialize PE arrays */
  void *pe_arrays[total_pe];
  for (int i = 0; i < total_pe; i++) {
    pe_arrays[i] = std::malloc(SIZE_PER_PE * ONE_BYTE);
  }
  /* Initialize input and weight */
  int input_size = HEIGHT * WIDTH * CHANNEL;
  float *input = (float *)std::malloc(input_size * sizeof(float));
  float *weight =
      (float *)std::malloc(OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE *
                           KERNEL_SIZE * sizeof(float));
  int weight_size =
      OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;
  for (int i = 0; i < input_size; i++) {
    input[i] = i;
  }
  for (int i = 0; i < weight_size; i++) {
    weight[i] = i;
  }
  // assume we are using first PE
  float *curr_pe = static_cast<float *>(pe_arrays[0]);
  // fill the PE with weight
  for (int i = 0; i < OUTPUT_CHANNELS; i++) {
    for (int j = 0; j < INPUT_CHANNELS; j++) {
      for (int k = 0; k < KERNEL_SIZE; k++) {
        for (int l = 0; l < KERNEL_SIZE; l++) {
          curr_pe[i * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE +
                  j * KERNEL_SIZE * KERNEL_SIZE + k * KERNEL_SIZE + l] =
              weight[i * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE +
                     j * KERNEL_SIZE * KERNEL_SIZE + k * KERNEL_SIZE + l];
        }
      }
    }
  }
  // followed by that fill the PE with input
  for (int i = 0; i < CHANNEL; i++) {
    for (int j = 0; j < HEIGHT / 2 + (OFFSET); j++) {
      for (int k = 0; k < WIDTH + (OFFSET); k++) {
        int jj = j - (KERNEL_SIZE / 2);
        int kk = k - (KERNEL_SIZE / 2);

        if (jj >= 0 && kk >= 0 && kk < WIDTH) {
          curr_pe[OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE +
                  i * (HEIGHT / 2 + (OFFSET)) * (WIDTH + (OFFSET)) +
                  j * (WIDTH + (OFFSET)) + k] =
              input[i * HEIGHT * WIDTH + jj * WIDTH + kk];
        } else {
          curr_pe[OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE +
                  i * (HEIGHT / 2 + (OFFSET)) * (WIDTH + (OFFSET)) +
                  j * (WIDTH + (OFFSET)) + k] = 0;
        }
      }
    }
  }

  float *second_pe = static_cast<float *>(pe_arrays[1]);
  // fill the rest of the input in second PE
  for (int i = 0; i < CHANNEL; i++) {
    for (int j = 0; j < HEIGHT / 2 + OFFSET; j++) {
      for (int k = 0; k < WIDTH + OFFSET; k++) {
        int jj = j + HEIGHT / 2 - (KERNEL_SIZE / 2);
        int kk = k - (KERNEL_SIZE / 2);

        if (jj >= 0 && jj < HEIGHT && kk >= 0 && kk < WIDTH) {
          second_pe[i * ((HEIGHT / 2) + OFFSET) * (WIDTH + OFFSET) +
                    j * (WIDTH + OFFSET) + k] =
              input[i * HEIGHT * WIDTH + jj * WIDTH + kk];
        } else {
          second_pe[i * ((HEIGHT / 2) + OFFSET) * (WIDTH + OFFSET) +
                    j * (WIDTH + OFFSET) + k] = 0;
        }
      }
    }
  }
  std::cout << "Initialized PE's" << std::endl;
  /* Declare and Allocate memory for input, weight, output tensors */
  const TensorShape input_shape(WIDTH + (OFFSET), (HEIGHT / 2) + (OFFSET),
                                INPUT_CHANNELS, 1);
  const TensorShape weight_shape(KERNEL_SIZE, KERNEL_SIZE, INPUT_CHANNELS,
                                 OUTPUT_CHANNELS);
  const TensorShape output_shape(WIDTH, (HEIGHT / 2), OUTPUT_CHANNELS, 1);
  Tensor input_tensor, weight_tensor, bias_tensor, output_tensor;
  input_tensor.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
  input_tensor.allocator()->allocate();
  weight_tensor.allocator()->init(TensorInfo(weight_shape, 1, DataType::F32));
  weight_tensor.allocator()->allocate();
  bias_tensor.allocator()->init(
      TensorInfo(TensorShape(OUTPUT_CHANNELS), 1, DataType::F32));
  bias_tensor.allocator()->allocate();
  output_tensor.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));
  output_tensor.allocator()->allocate();
  /* copy input and weight from PE into Tensors */
  std::memcpy(
      input_tensor.buffer(),
      curr_pe + (KERNEL_SIZE * KERNEL_SIZE * INPUT_CHANNELS * OUTPUT_CHANNELS),
      (INPUT_CHANNELS * ((HEIGHT / 2) + (OFFSET)) * (WIDTH + (OFFSET))) *
          sizeof(float));
  save_tensor_to_npy(input_tensor, "new_test/input/cpp_input_1.npy");
  std::cout << "input done shape : "
            << get_tensor_shape(input_tensor.info()->tensor_shape())
            << std::endl;
  std::memcpy(weight_tensor.buffer(), curr_pe,
              KERNEL_SIZE * KERNEL_SIZE * 3 * OUTPUT_CHANNELS * sizeof(float));
  std::cout << "weigth done shape : "
            << get_tensor_shape(input_tensor.info()->tensor_shape())
            << std::endl;
  save_tensor_to_npy(weight_tensor, "new_test/weight/cpp_weight.npy");
  std::memset(bias_tensor.buffer(), 0, OUTPUT_CHANNELS * sizeof(float));
  std::cout << "Initialized tensors" << std::endl;

  /* Perform convolution */
  NEConvolutionLayer conv;
  conv.configure(&input_tensor, &weight_tensor, &bias_tensor, &output_tensor,
                 PadStrideInfo(1, 1, 0, 0));
  conv.run();

  std::cout << "conv done shape "
            << get_tensor_shape(output_tensor.info()->tensor_shape())
            << std::endl;
  save_tensor_to_npy(output_tensor, "new_test/output/cpp_output_wtpad_1.npy");

  /* Declare and Allocate memory for input, output tensors for the second half.
   * weights can be reused */
  Tensor input_tensor_2, output_tensor_2;
  input_tensor_2.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
  input_tensor_2.allocator()->allocate();
  output_tensor_2.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));
  output_tensor_2.allocator()->allocate();
  /*Copy input and weight from PE into Tensors*/
  std::memcpy(
      input_tensor_2.buffer(), second_pe,
      (INPUT_CHANNELS * ((HEIGHT / 2) + (OFFSET)) * (WIDTH + (OFFSET))) *
          sizeof(float));
  save_tensor_to_npy(input_tensor_2, "new_test/input/cpp_input_2.npy");
  std::cout << "input done shape : "
            << get_tensor_shape(input_tensor_2.info()->tensor_shape())
            << std::endl;
  std::cout << "Initialized tensors" << std::endl;
  /* Perform convolution */
  NEConvolutionLayer conv_2;
  conv_2.configure(&input_tensor_2, &weight_tensor, &bias_tensor,
                   &output_tensor_2, PadStrideInfo(1, 1, 0, 0));
  conv_2.run();
  std::cout << "conv done Output shape "
            << get_tensor_shape(output_tensor_2.info()->tensor_shape())
            << std::endl;
  save_tensor_to_npy(output_tensor_2, "new_test/output/cpp_output_wtpad_2.npy");

  /* merge curr_pe and second_pe into one */
  float *merged_pe =
      (float *)std::malloc(HEIGHT * OUTPUT_CHANNELS * WIDTH * sizeof(float));
  float *output_1_buffer = (float *)output_tensor.buffer();
  float *output_2_buffer = (float *)output_tensor_2.buffer();
  for (int i = 0; i < OUTPUT_CHANNELS; i++) {
    for (int j = 0; j < (HEIGHT / 2); j++) {
      for (int k = 0; k < WIDTH; k++) {
        // Copy from output_1_buffer to the first half of merged_pe
        merged_pe[i * HEIGHT * WIDTH + j * WIDTH + k] =
            output_1_buffer[i * ((HEIGHT / 2) * WIDTH) + j * WIDTH + k];
        // Copy from output_2_buffer to the second half of merged_pe
        merged_pe[i * HEIGHT * WIDTH + (j + HEIGHT / 2) * WIDTH + k] =
            output_2_buffer[i * ((HEIGHT / 2) * WIDTH) + j * WIDTH + k];
      }
    }
  }
  /* Merge the outputs into one */
  TensorShape merged_output_shape(WIDTH, HEIGHT, OUTPUT_CHANNELS, 1);
  Tensor merged_output_tensor;
  merged_output_tensor.allocator()->init(
      TensorInfo(merged_output_shape, 1, DataType::F32));
  merged_output_tensor.allocator()->allocate();
  std::memcpy(merged_output_tensor.buffer(), merged_pe,
              (WIDTH * (HEIGHT)*OUTPUT_CHANNELS * 1) * sizeof(float));
  save_tensor_to_npy(merged_output_tensor,
                     "new_test/output/cpp_output_wtpad_merged.npy");
  std::cout << "merged output done shape : "
            << get_tensor_shape(merged_output_tensor.info()->tensor_shape())
            << std::endl;
  return 0;
}