#include <omp.h>
#include <x86intrin.h>

#include "compute.h"

//easier to create new matrix
void flip_matrix(matrix_t* b_matrix, matrix_t* flipped_matrix) {

    flipped_matrix->rows = b_matrix->rows;
    flipped_matrix->cols = b_matrix->cols;
    flipped_matrix->data = (int32_t*)malloc(b_matrix->rows * b_matrix->cols * sizeof(int32_t));

    for (uint32_t i = 0; i < b_matrix->rows; i++) {
        for (uint32_t j = 0; j < b_matrix->cols; j++) {            
            flipped_matrix->data[(b_matrix->rows - 1 - i) * b_matrix->cols + b_matrix->cols - 1 - j] =
                b_matrix->data[i * b_matrix->cols + j];
        }
    }
}

int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {


  if (!a_matrix || !b_matrix || !output_matrix) {
      return -1;
  }
  uint32_t output_rows = a_matrix->rows - b_matrix->rows + 1;
  uint32_t output_cols = a_matrix->cols - b_matrix->cols + 1;
  *output_matrix = (matrix_t*)malloc(sizeof(matrix_t));

  if (!(*output_matrix)) {
      return -1;
  }

  (*output_matrix)->rows = output_rows;
  (*output_matrix)->cols = output_cols;

  (*output_matrix)->data = (int32_t*)malloc(output_rows * output_cols * sizeof(int32_t));
  if (!(*output_matrix)->data) {
      free(*output_matrix);
      return -1;
  }

  matrix_t flipped_b_matrix;
  flip_matrix(b_matrix, &flipped_b_matrix);

  #pragma omp parallel for collapse(2)
  for (uint32_t i = 0; i < output_rows; i++) {
      for (uint32_t j = 0; j < output_cols; j++) {
          __m256i sum_vec = _mm256_setzero_si256(); 
          int32_t conv_sum = 0;
          for (uint32_t m = 0; m < flipped_b_matrix.rows; m++) {
              for (uint32_t n = 0; n < flipped_b_matrix.cols / 8 * 8; n += 8) {
                  __m256i a = _mm256_loadu_si256((__m256i*)&a_matrix->data[(i + m) * a_matrix->cols + j + n]);
                  __m256i b = _mm256_loadu_si256((__m256i*)&flipped_b_matrix.data[m * flipped_b_matrix.cols + n]);

                  __m256i product_vec = _mm256_mullo_epi32(a, b);
                  sum_vec = _mm256_add_epi32(sum_vec, product_vec);
              }
              //tail condition
              for (uint32_t n = (flipped_b_matrix.cols/8) * 8; n < flipped_b_matrix.cols; n++) {
                  conv_sum += a_matrix->data[(i + m) * a_matrix->cols + j + n] * flipped_b_matrix.data[m * flipped_b_matrix.cols+ n];
              }
          }
          int32_t tmp[8];
          _mm256_storeu_si256((__m256i*) tmp, sum_vec);
          conv_sum += (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]+tmp[5]+tmp[6]+tmp[7]);
          (*output_matrix)->data[j + i * output_cols] = conv_sum;
      }
  }

  free(flipped_b_matrix.data);


  return 0;
}

// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  char *a_matrix_path = get_a_matrix_path(task);
  if (read_matrix(a_matrix_path, &a_matrix)) {
    printf("Error reading matrix from %s\n", a_matrix_path);
    return -1;
  }
  free(a_matrix_path);

  char *b_matrix_path = get_b_matrix_path(task);
  if (read_matrix(b_matrix_path, &b_matrix)) {
    printf("Error reading matrix from %s\n", b_matrix_path);
    return -1;
  }
  free(b_matrix_path);

  if (convolve(a_matrix, b_matrix, &output_matrix)) {
    printf("convolve returned a non-zero integer\n");
    return -1;
  }

  char *output_matrix_path = get_output_matrix_path(task);
  if (write_matrix(output_matrix_path, output_matrix)) {
    printf("Error writing matrix to %s\n", output_matrix_path);
    return -1;
  }
  free(output_matrix_path);

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
