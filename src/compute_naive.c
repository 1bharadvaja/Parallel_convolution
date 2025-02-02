#include "compute.h"

// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  
  if (!a_matrix || !b_matrix || !output_matrix) {
        return -1;
    } 
  if (a_matrix->rows < b_matrix->rows || a_matrix->cols < b_matrix->cols) {
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

  for (uint32_t i = 0; i < output_rows; i++) {
    for (uint32_t j = 0; j < output_cols; j++) {
        int32_t sum = 0;
        for (uint32_t m = 0; m < b_matrix->rows; m++) {
            for (uint32_t n = 0; n < b_matrix->cols; n++) {
                // Calculate flipped index for b_matrix
                uint32_t b_row = b_matrix->rows - 1 - m;
                uint32_t b_col = b_matrix->cols - 1 - n;

                // Get indices in a_matrix
                uint32_t a_row = i + m;
                uint32_t a_col = j + n;

                // Element-wise multiplication and sum
                sum += a_matrix->data[a_row * a_matrix->cols + a_col] * 
                        b_matrix->data[b_row * b_matrix->cols + b_col];
            }
        }
        // Store result in the output matrix
        (*output_matrix)->data[i * output_cols + j] = sum;
    }
  }

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
